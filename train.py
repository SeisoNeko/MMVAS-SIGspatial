import os
import random
import logging
import sys
import optuna
import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph
from torchvision.models import ResNet50_Weights
from transformers import AutoModel
import matplotlib.pyplot as plt
import yaml

from models.GridLearner import GridLearner
from models.AdaRegionGen import AdaRegionGen
from models.DownStreamTask import DownstreamTaskModel
from models.loss import FusionSimilarityLoss, CustomMSELoss

from dataset.mmurgdataset import MMURGDataContainer, collate_fn
import gc

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()
OUTPUT_DIR = config["paths"]["output_weights_dir"]
CACHE_DIR = config["paths"]["cache_dir"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config["paths"]["log_dir"] + "/train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_subgraph_batch(node_idx, edge_index, edge_attr=None, num_nodes=None):
    """Helper function to extract a subgraph for memory-safe GNN training."""
    sub_edge_index, sub_edge_attr = subgraph(
        node_idx, edge_index, edge_attr, relabel_nodes=True, num_nodes=num_nodes
    )
    return sub_edge_index, sub_edge_attr


def save_training_curves(train_losses, val_losses, output_path, title):
    """Save train/validation loss curves to an image file."""
    if not train_losses and not val_losses:
        return

    plt.figure(figsize=(10, 6))
    start_idx = min(10, len(train_losses) - 1)
    if train_losses:
        plt.plot(range(start_idx + 1, len(train_losses) + 1), train_losses[start_idx:], label="Train Loss", linewidth=2)
    if val_losses:
        plt.plot(range(start_idx + 1, len(val_losses) + 1), val_losses[start_idx:], label="Validation Loss", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def train():
    # Ensure weights directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Hardware Configuration
    device = torch.device(config["global"]["device"] if torch.cuda.is_available() else "cpu")
    logging.info(f"Training on device: {device} | Grid Type: 100m")
    
    # 2. Strict Hyperparameters based on MMURG experimental settings
    embed_dim = config["model"]["embed_dim"]
    
    # GridLearner Hyperparameters
    grid_lr = config["training"]["grid_learner"]["lr"]
    grid_epochs = config["training"]["grid_learner"]["epochs"]
    grid_batch_size = config["training"]["grid_learner"]["batch_size"]
    
    # Region Enricher Hyperparameters
    region_lr = config["training"]["region_enricher"]["lr"]
    weight_decay = config["training"]["region_enricher"]["weight_decay"]
    lambda_weight = config["training"]["loss"]["lambda"]
    region_epochs = config["training"]["region_enricher"]["epochs"]
    patience = config["training"]["patience"]
    dropout_rate = config["training"]["region_enricher"]["dropout_rate"]
    set_seed(config["global"]["seed"])
    
    # log all parameters
    logging.info("\n--- Hyperparameters ---")
    logging.info(f"GridLearner Learning Rate: {grid_lr}")
    logging.info(f"GridLearner Epochs: {grid_epochs}")
    logging.info(f"Region Enricher Learning Rate: {region_lr}")
    logging.info(f"Region Enricher Weight Decay: {weight_decay}")
    logging.info(f"Region Enricher Epochs: {region_epochs}")
    logging.info(f"Early Stopping Patience: {patience}")
    logging.info(f"Embedding Dimension: {embed_dim}")
    logging.info(f"Random Seed: {config['global']['seed']}")
    logging.info(f"Dropout Rate: {dropout_rate}")

    # 3. Load Preprocessed Datasets
    data = MMURGDataContainer(
        config = config
    )
    m_cells = data.cell_dataset.num_cells
    
    total_regions = len(data.region_dataset)
    train_size = int(0.7 * total_regions)
    val_size = int(0.15 * total_regions)
    test_size = total_regions - train_size - val_size
    
    logging.info(f"\nPhase 3 Data Split: {train_size} Train | {val_size} Val | {test_size} Test")
    
    # Fix the generator seed for reproducible splits across runs
    generator = torch.Generator().manual_seed(config["global"]["seed"])
    train_dataset, val_dataset, test_dataset = random_split(
        data.region_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=True, num_workers=config["dataloader"]["num_workers"], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=False, num_workers=config["dataloader"]["num_workers"], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=False, num_workers=config["dataloader"]["num_workers"], collate_fn=collate_fn)
    
    # 4. Initialize the Models
    input_dims = {
        'poi': data.cell_dataset.poi_feat.shape[1],
        'land_use': data.cell_dataset.lu_feat.shape[1],
        'neighbor': data.cell_dataset.gn_feat.shape[1],
        'farbac': data.cell_dataset.farbac_feat.shape[1]
    }
    
    grid_learner = GridLearner(input_dims=input_dims, embed_dim=embed_dim).to(device)
    ada_region_gen = AdaRegionGen() # Deterministic, no device placement needed
    task_model = DownstreamTaskModel(embed_dim=embed_dim, dropout_rate=dropout_rate).to(device)
    
    # 5. Optimizers
    grid_optimizer = optim.Adam(grid_learner.parameters(), lr=grid_lr)
    task_optimizer = optim.Adam(task_model.parameters(), lr=region_lr, weight_decay=weight_decay)

    # 6. Loss Functions
    similarity_criterion = FusionSimilarityLoss().to(device)
    region_criterion = nn.MSELoss().to(device)
    grid_criterion = CustomMSELoss(z_threshold = config["training"]["loss"]["z_threshold"], tail_weight=config["training"]["loss"]["tail_weight"]).to(device)

    # -------------------------------------------------------------------------
    # PHASE 0: Offline Text & SVI Extraction (Preventing Memory Bottlenecks)
    # -------------------------------------------------------------------------
    logging.info("\n--- Phase 0: Extracting LLM Embeddings ---")

    sat_cache_path = os.path.join(CACHE_DIR, config["paths"]["sat_cache_name"])
    text_cache_path = os.path.join(CACHE_DIR, config["paths"]["text_cache_name"])

    # Text Extraction
    if os.path.exists(text_cache_path) and not config["pipeline"]["refresh_data"]:
        logging.info("Loading cached LLM Text Embeddings...")
        H_t = torch.load(text_cache_path, map_location=device, weights_only=True)
    else:
        logging.info("Extracting LLM Text Embeddings (This will run once)...")
        llm = AutoModel.from_pretrained("bert-base-chinese").to(device).eval()
        
        all_input_ids = data.region_dataset.text_input_ids.to(device) 
        all_attention_mask = data.region_dataset.text_attention_mask.to(device)
        
        with torch.no_grad():
            outputs = llm(input_ids=all_input_ids, attention_mask=all_attention_mask)
            H_t = outputs.last_hidden_state[:, 0, :] 
            
        torch.save(H_t, text_cache_path)
        del llm
        torch.cuda.empty_cache()

    grid_to_region_mapping = data.grid_to_region_mapping.to(device) 
    text_emb_mapped = H_t[grid_to_region_mapping] 

    # Satellite Extraction
    if os.path.exists(sat_cache_path) and not config["pipeline"]["refresh_data"]:
        logging.info("Loading cached ResNet50 Satellite Features...")
        full_sat_features = torch.load(sat_cache_path, map_location=device, weights_only=True)
    else:
        logging.info(f"Extracting {m_cells} Satellite Image Features (This will run once)...")
        seq_cell_loader = DataLoader(data.cell_dataset, batch_size=config["dataloader"]["satellite"]["batch_size"], shuffle=False, num_workers=config["dataloader"]["satellite"]["num_workers"]) 
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        resnet.fc = nn.Identity()
        resnet.eval() 
        
        all_sat_features = []
        with torch.no_grad():
            for batch in tqdm.tqdm(seq_cell_loader, desc="Processing Satellite Images"):
                imgs = batch['sat_image'].to(device)
                features = resnet(imgs)
                all_sat_features.append(features.cpu()) 
                
        full_sat_features = torch.cat(all_sat_features, dim=0)
        torch.save(full_sat_features, sat_cache_path)
        del resnet
        gc.collect()
        torch.cuda.empty_cache()
        
    full_sat_features = full_sat_features.to(device)

    # =====================================================================
    # PHASE 1: Train GridLearner (Self-Supervised Multi-Task)
    # =====================================================================
    logging.info("\n--- Starting Phase 1: GridLearner Training ---")
    grid_learner.train()
    
    poi_feat = data.cell_dataset.poi_feat.to(device)
    poi_edge_index = data.poi_edge_index.to(device)
    poi_edge_weights = data.poi_edge_weights.to(device)
    
    lu_feat = data.cell_dataset.lu_feat.to(device)
    lu_edge_index = data.lu_edge_index.to(device)
    lu_edge_weights = data.lu_edge_weights.to(device)
    
    gn_feat = data.cell_dataset.gn_feat.to(device)
    gn_edge_index = data.gn_edge_index.to(device)
    
    farbac_feat = data.cell_dataset.farbac_feat.to(device)
    farbac_edge_index = data.farbac_edge_index.to(device)
    farbac_edge_weights = data.farbac_edge_weights.to(device)
    
    if config["pipeline"]["p1_skip"]:
        logging.info("\n--- [Phase 1 Skipped] Loading pre-trained E ---")
        final_E = torch.load(f'{OUTPUT_DIR}/phase1_embeddings_E.pt', map_location=device, weights_only=True)
    else:
        logging.info("\n--- Starting Phase 1: Subgraph Batched GridLearner ---")
        grid_learner.train()

        for epoch in range(grid_epochs):
            permuted_nodes = torch.randperm(m_cells, device=device)
            epoch_loss = 0.0
            best_loss = float('inf')
            grid_epoch_no_improve = 0
            
            for i in range(0, m_cells, grid_batch_size):
                grid_optimizer.zero_grad()
                b_nodes = permuted_nodes[i:i+grid_batch_size]
                
                # 1. Slice Node Features
                b_poi_f, b_lu_f, b_gn_f, b_farbac_f = poi_feat[b_nodes], lu_feat[b_nodes], gn_feat[b_nodes], farbac_feat[b_nodes]
                b_sat, b_text = full_sat_features[b_nodes], text_emb_mapped[b_nodes]
                
                # 2. Extract Subgraphs mapping edges only to the current batch of nodes
                b_poi_e, b_poi_w = get_subgraph_batch(b_nodes, poi_edge_index, poi_edge_weights, num_nodes=m_cells)
                b_lu_e, b_lu_w = get_subgraph_batch(b_nodes, lu_edge_index, lu_edge_weights, num_nodes=m_cells)
                b_gn_e, _ = get_subgraph_batch(b_nodes, gn_edge_index, num_nodes=m_cells)
                b_farbac_e, b_farbac_w = get_subgraph_batch(b_nodes, farbac_edge_index, farbac_edge_weights, num_nodes=m_cells)
                
                E_batch, intra_views = grid_learner(
                    b_poi_f, b_poi_e, b_poi_w,
                    b_lu_f, b_lu_e, b_lu_w,
                    b_gn_f, b_gn_e, 
                    b_farbac_f, b_farbac_e, b_farbac_w,
                    b_sat, b_text
                )
                
                loss = similarity_criterion(intra_views, E_batch)
                loss.backward()
                grid_optimizer.step()
                epoch_loss += loss.item()
                
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / (m_cells // grid_batch_size + 1)
                logging.info(f"GridLearner Epoch [{epoch+1}/{grid_epochs}] | Avg Subgraph Loss: {avg_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                grid_epoch_no_improve = 0
            else:
                grid_epoch_no_improve += 1

            if grid_epoch_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch+1} with best loss {best_loss:.4f}")
                break
                
        logging.info("Generating full-city Grid Embeddings (Chunked)...")
        grid_learner.eval()
        final_E = torch.zeros((m_cells, embed_dim), device=device)
        with torch.no_grad():
            node_seq = torch.arange(m_cells, device=device)
            for i in range(0, m_cells, grid_batch_size):
                b_nodes = node_seq[i:i+grid_batch_size]
                
                b_poi_e, b_poi_w = get_subgraph_batch(b_nodes, poi_edge_index, poi_edge_weights, num_nodes=m_cells)
                b_lu_e, b_lu_w = get_subgraph_batch(b_nodes, lu_edge_index, lu_edge_weights, num_nodes=m_cells)
                b_gn_e, _ = get_subgraph_batch(b_nodes, gn_edge_index, num_nodes=m_cells)
                b_farbac_e, b_farbac_w = get_subgraph_batch(b_nodes, farbac_edge_index, farbac_edge_weights, num_nodes=m_cells)
                
                E_batch, _ = grid_learner(
                    poi_feat[b_nodes], b_poi_e, b_poi_w,
                    lu_feat[b_nodes], b_lu_e, b_lu_w,
                    gn_feat[b_nodes], b_gn_e, 
                    farbac_feat[b_nodes], b_farbac_e, b_farbac_w,
                    full_sat_features[b_nodes], text_emb_mapped[b_nodes]
                )
                final_E[b_nodes] = E_batch

        torch.save(final_E.cpu(), f'{OUTPUT_DIR}/phase1_embeddings_E.pt')
        torch.save(grid_learner.state_dict(), f'{OUTPUT_DIR}/grid_learner_final.pth')
        logging.info("Phase 1 Grid Embeddings and Model Weights saved to disk.")

        del grid_learner
        del grid_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Phase 1 GPU VRAM successfully released.")
        
    # =====================================================================
    # PHASE 2: AdaRegionGen (Deterministic Geometric Projection)
    # =====================================================================
    logging.info("\n--- Starting Phase 2: Adaptive Region Generation ---")
    H = ada_region_gen.forward(
        regions_gdf=data.regions_gdf, 
        cells_gdf=data.cells_gdf, 
        cell_embeddings=final_E
    )
    H = H.to(device)
    logging.info(f"Generated Region Embedding Matrix H of shape: {H.shape}")
    
    # =====================================================================
    # PHASE 3: Train Region Enricher (Supervised Downstream Task)
    # =====================================================================
    if config["pipeline"]["tune"]:
        logging.info("\n--- Starting Optuna Hyperparameter Tuning for Phase 3 ---")
        
        def objective(trial):
            # 1. Suggest Hyperparameters
            trial_lr = trial.suggest_float("region_lr", 1e-5, 1e-2, log=True)
            trial_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
            trial_lambda = trial.suggest_float("lambda", 0.01, 100.0, log=True)
            trial_dropout_rate = trial.suggest_float("dropout_rate", 1e-5, 0.5, log=True)
            
            # 2. Initialize a fresh model and optimizer for this trial
            trial_model = DownstreamTaskModel(embed_dim=embed_dim, dropout_rate=trial_dropout_rate).to(device)
            trial_optimizer = optim.Adam(trial_model.parameters(), lr=trial_lr, weight_decay=trial_weight_decay)
            trial_grid_criterion = CustomMSELoss(z_threshold = config["training"]["loss"]["z_threshold"], tail_weight=config["training"]["loss"]["tail_weight"]).to(device)
            
            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            for epoch in range(region_epochs):
                # --- TRAINING PASS ---
                trial_model.train()
                for batch in train_loader:
                    trial_optimizer.zero_grad()
                    
                    batch_region_idx = batch["region_idx"].to(device)
                    sv_images = batch["street_view_images"].to(device)
                    raw_tax = batch["raw_tax_sequence"].to(device)
                    target_grid_indices = batch["grid_indices_in_region"].to(device) 
                    target_grid_labels = batch["grid_land_values"].to(device) 
                    target_region_avgs = batch["region_avg_value"].to(device)
                    grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
                    
                    batch_H = H[batch_region_idx]

                    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                        node_idx=target_grid_indices,
                        num_hops=2,
                        edge_index=gn_edge_index,
                        relabel_nodes=True, 
                        num_nodes=len(final_E)
                    )
                    b_E_grid = final_E[subset]
                    
                    predictions, region_base = trial_model(
                        E_grid=b_E_grid, 
                        grid_edge_index=sub_edge_index, 
                        batch_H=batch_H, 
                        sv_images=sv_images, 
                        raw_tax_sequences=raw_tax, 
                        target_indices=mapping,
                        grid_to_batch_idx=grid_to_batch_idx
                    )
                    
                    region_loss = region_criterion(region_base, target_region_avgs)
                    grid_loss = trial_grid_criterion(predictions, target_grid_labels.view(-1))
                    loss = region_loss + (trial_lambda * grid_loss)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trial_model.parameters(), max_norm=1.0)
                    trial_optimizer.step()
                
                # --- VALIDATION PASS ---
                trial_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch_region_idx = batch["region_idx"].to(device)
                        sv_images = batch["street_view_images"].to(device)
                        raw_tax = batch["raw_tax_sequence"].to(device)
                        target_grid_indices = batch["grid_indices_in_region"].to(device) 
                        target_grid_labels = batch["grid_land_values"].to(device)
                        target_region_avgs = batch["region_avg_value"].to(device)
                        grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
                        batch_H = H[batch_region_idx]
                        
                        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                            node_idx=target_grid_indices,
                            num_hops=2,
                            edge_index=gn_edge_index,
                            relabel_nodes=True, 
                            num_nodes=len(final_E)
                        )
                        b_E_grid = final_E[subset]
                        
                        predictions, region_base = trial_model(
                            E_grid=b_E_grid, 
                            grid_edge_index=sub_edge_index, 
                            batch_H=batch_H, 
                            sv_images=sv_images, 
                            raw_tax_sequences=raw_tax, 
                            target_indices=mapping,
                            grid_to_batch_idx=grid_to_batch_idx
                        )
                        region_loss = region_criterion(region_base, target_region_avgs)
                        grid_loss = trial_grid_criterion(predictions, target_grid_labels.view(-1))
                        loss = region_loss + grid_loss
                        val_loss += loss.item()
                        
                avg_val_loss = val_loss / len(val_loader)
                
                # Update best loss and early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                # Report intermediate values to Optuna
                trial.report(avg_val_loss, epoch)
                
                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                    
                if epochs_no_improve >= patience:
                    break
                    
            return best_val_loss

        # Create and run the Optuna study
        study = optuna.create_study(direction="minimize", 
                                    study_name=config["training"]["optuna"]["study_name"],
                                    storage="sqlite:///tuning_history.db",
                                    load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=config["training"]["optuna"]["n_startup_trials"],
                                        n_warmup_steps=config["training"]["optuna"]["n_warmup_steps"],
                                        interval_steps=config["training"]["optuna"]["interval_steps"]
                                    ))
        logging.info("Starting optimization. Press Ctrl+C to stop early and keep best results.")
        
        try:
            # You can adjust n_trials to however many combinations you want to test
            study.optimize(objective, n_trials=config["training"]["optuna"]["n_trials"], timeout=None) 
        except KeyboardInterrupt:
            logging.info("Optimization interrupted by user. Returning best trial so far.")

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        logging.info("Study statistics: ")
        logging.info(f"  Number of finished trials: {len(study.trials)}")
        logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
        logging.info(f"  Number of complete trials: {len(complete_trials)}")

        logging.info("Best trial:")
        trial = study.best_trial
        logging.info(f"  Value (Validation MSE): {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")
            
        logging.info("Optuna tuning complete. Please update your command-line arguments with the best parameters and re-run without --tune.")
        return # Exit after tuning, so we don't accidentally run Phase 4 with an untrained model
    else:
        logging.info("\n--- Starting Phase 3: Region Enricher Task Training ---")
        
        best_val_loss = float('inf')
        best_model_path = f'{OUTPUT_DIR}/best_task_model.pth'
        epochs_no_improve = 0
        train_loss_history = []
        val_loss_history = []
        
        final_E = final_E.to(device)
        gn_edge_index = data.gn_edge_index.to(device)

        for epoch in range(region_epochs):
            task_model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                task_optimizer.zero_grad()
                
                batch_region_idx = batch["region_idx"].to(device)
                sv_images = batch["street_view_images"].to(device)
                raw_tax = batch["raw_tax_sequence"].to(device)
                
                target_grid_indices = batch["grid_indices_in_region"].to(device) 
                target_grid_labels = batch["grid_land_values"].to(device)
                target_region_avgs = batch["region_avg_value"].to(device)
                grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
                
                batch_H = H[batch_region_idx]

                subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx=target_grid_indices,
                    num_hops=2,
                    edge_index=gn_edge_index,
                    relabel_nodes=True, 
                    num_nodes=len(final_E)
                )
                b_E_grid = final_E[subset]
                
                final_preds, region_base = task_model(
                    E_grid=b_E_grid, 
                    grid_edge_index=sub_edge_index, 
                    batch_H=batch_H, 
                    sv_images=sv_images, 
                    raw_tax_sequences=raw_tax, 
                    target_indices=mapping,
                    grid_to_batch_idx=grid_to_batch_idx
                )
                
                loss_region = region_criterion(region_base, target_region_avgs)
                loss_grid = grid_criterion(final_preds, target_grid_labels.view(-1))
                loss = loss_region + (lambda_weight * loss_grid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(task_model.parameters(), max_norm=1.0)
                task_optimizer.step()
                
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)
            
            # --- VALIDATION PASS ---
            task_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_region_idx = batch["region_idx"].to(device)
                    sv_images = batch["street_view_images"].to(device)
                    raw_tax = batch["raw_tax_sequence"].to(device)
                    
                    target_grid_indices = batch["grid_indices_in_region"].to(device) 
                    target_grid_labels = batch["grid_land_values"].to(device)
                    target_region_avgs = batch["region_avg_value"].to(device)
                    grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
                    
                    batch_H = H[batch_region_idx]

                    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                        node_idx=target_grid_indices,
                        num_hops=2,
                        edge_index=gn_edge_index,
                        relabel_nodes=True, 
                        num_nodes=len(final_E)
                    )
                    b_E_grid = final_E[subset]

                    final_preds, region_base = task_model(
                        E_grid=b_E_grid,
                        grid_edge_index=sub_edge_index, 
                        batch_H=batch_H, 
                        sv_images=sv_images, 
                        raw_tax_sequences=raw_tax, 
                        target_indices=mapping,
                        grid_to_batch_idx=grid_to_batch_idx
                    )
                    
                    loss_region = region_criterion(region_base, target_region_avgs)
                    loss_grid = grid_criterion(final_preds, target_grid_labels.view(-1))
                    loss = loss_region + loss_grid
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            
            # --- CHECKPOINT SAVING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(task_model.state_dict(), best_model_path)
                epochs_no_improve = 0
                is_new_best = "*"
            else:
                epochs_no_improve += 1
                is_new_best = ""
                
            if (epoch + 1) % 10 == 0:
                logging.info(f"Task Model Epoch [{epoch+1}/{region_epochs}] | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f} {is_new_best}")
                
            if epochs_no_improve >= patience:
                logging.info(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best weights.")
                break

        curves_path = os.path.join(OUTPUT_DIR, "phase3_training_curves.png")
        save_training_curves(
            train_loss_history,
            val_loss_history,
            curves_path,
            "Phase 3 Training Curves"
        )
        logging.info(f"Saved training curve image to {curves_path}")

        logging.info(f"\nPhase 3 Training Complete. Best Validation MSE: {best_val_loss:.4f}")

        # =====================================================================
        # PHASE 4: Final Test Evaluation
        # =====================================================================
        logging.info("\n--- Starting Phase 4: Final Test Evaluation ---")
        
        # Load the best weights we saved during validation
        task_model.load_state_dict(torch.load(best_model_path, map_location=device))
        task_model.eval()

        label_mean = data.region_dataset.label_mean.to(device)
        label_std = data.region_dataset.label_std.to(device)
        
        # Pre-extract geographic ID mapping for fast logging
        node_to_geo_id = dict(zip(data.cells_gdf['node_id'], data.cells_gdf['id']))
        
        logging.info("\n" + "="*80)
        logging.info(f"{'Node ID':<10} | {'Geo ID':<10} | {'Region ID':<10} | {'Real Value':<12} | {'Predicted':<12} | {'Abs Error':<12}")
        logging.info("="*80)
        
        all_real = []
        all_pred = []
        all_grid_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_region_idx = batch["region_idx"].to(device)
                sv_images = batch["street_view_images"].to(device)
                raw_tax = batch["raw_tax_sequence"].to(device)
                
                target_grid_indices = batch["grid_indices_in_region"].to(device) 
                labels_norm = batch["grid_land_values"].to(device)
                grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
                
                batch_H = H[batch_region_idx]

                subset, sub_edge_index, mapping, _ = k_hop_subgraph(
                        node_idx=target_grid_indices,
                        num_hops=2,
                        edge_index=gn_edge_index,
                        relabel_nodes=True, 
                        num_nodes=len(final_E)
                )
                b_E_grid = final_E[subset]
                
                # Get normalized predictions
                final_preds_norm, _ = task_model(
                    E_grid=b_E_grid, 
                    grid_edge_index=sub_edge_index, 
                    batch_H=batch_H, 
                    sv_images=sv_images, 
                    raw_tax_sequences=raw_tax,
                    target_indices=mapping,
                    grid_to_batch_idx=grid_to_batch_idx
                )
                
                # --- UN-NORMALIZE ---
                # Convert Z-scores back into actual land value rates
                real_values = (labels_norm.view(-1) * label_std) + label_mean
                predicted_values = (final_preds_norm * label_std) + label_mean
                
                real_np = real_values.cpu().numpy()
                pred_np = predicted_values.cpu().numpy()
                
                # THE FIX: Ensure indices are flattened before converting to numpy
                grids_np = target_grid_indices.view(-1).cpu().numpy()
                
                all_real.extend(real_np)
                all_pred.extend(pred_np)
                all_grid_ids.extend(grids_np)
                
                # Print a few samples from each batch to inspect during training
                sample_size = min(3, len(grids_np)) 
                for i in range(sample_size):
                    g_idx = grids_np[i]
                    r_idx = data.grid_to_region_mapping[g_idx].item()
                    geo_id = node_to_geo_id.get(g_idx, "N/A")
                    
                    r_val = real_np[i]
                    p_val = pred_np[i]
                    err = abs(r_val - p_val)
                    logging.info(f"{g_idx:<10} | {geo_id:<10} | {r_idx:<10} | {r_val:<12.4f} | {p_val:<12.4f} | {err:<12.4f}")

        # Calculate Final Real-World Metrics
        all_real = np.array(all_real)
        all_pred = np.array(all_pred)
        all_errs = all_pred - all_real
        all_errs_abs = np.abs(all_errs)
        
        mae = np.mean(all_errs_abs)
        mse = np.mean(all_errs**2)
        rmse = np.sqrt(mse)
        
        logging.info("="*80)
        logging.info(f"FINAL TEST METRICS (Un-Normalized / Real Units)")
        logging.info(f"Total Test Grids Evaluated: {len(all_real)}")
        logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
        logging.info(f"Mean Squared Error (MSE):  {mse:.4f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logging.info("="*80)

        # Merge correctly using the perfectly aligned node_ids
        results_df = pd.DataFrame({
            'node_id': all_grid_ids,
            'real_value': all_real,
            'predicted_value': all_pred,
            'error': all_errs,
            'abs_error': all_errs_abs
        })
        
        # The inner merge cleanly attaches predictions to the correct polygons
        merged_gdf = data.cells_gdf.merge(results_df, on='node_id', how='inner')
        
        os.makedirs("./result", exist_ok=True)
        
        # Export to CSV
        csv_path = "./result/test_predictions.csv"
        merged_gdf.to_csv(csv_path, index=False)
        
        # Export to GPKG for direct QGIS usage
        gpkg_path = "./result/test_predictions.gpkg"
        merged_gdf.to_file(gpkg_path, driver="GPKG")
        
        logging.info(f"Spatial predictions saved to {csv_path} and {gpkg_path}")
        logging.info("Pipeline fully complete. Model is ready for deployment.")

if __name__ == "__main__":
    train()