import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import sys
import pandas as pd
import os
sys.path.append(os.getcwd())

from models.AdaRegionGen import AdaRegionGen
from models.RegionEnricher import RegionEnricher
from models.LocalGridPredictor import LocalGridPredictor
from dataset.mmurgdataset import MMURGDataContainer, collate_fn
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DownstreamTaskModel(nn.Module):
    """
    Wraps Phase 3 & 4 for inference, utilizing the VRAM-optimized target_indices.
    """
    def __init__(self, args):
        super().__init__()
        self.enricher = RegionEnricher(d=args.embed_dim, d_img=2048, raw_tax_dim=3, d_tax=128, d_proj=256)
        self.predictor = LocalGridPredictor(d=args.embed_dim)

    def forward(self, E_grid, grid_edge_index, batch_H, sv_images, raw_tax_sequences, target_indices):
        H_enriched = self.enricher(batch_H, sv_images, raw_tax_sequences)
        grid_preds = self.predictor(E_grid, grid_edge_index, H_enriched, target_indices)
        return grid_preds

def run_inference(args):
    weights_dir = f"./weights/{args.version}" if args.version else "./weights"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logging.info(f"Running Inference on device: {device}")
    
    # 1. Load the Dataset
    logging.info("\nLoading Data Container...")
    data = MMURGDataContainer(
        npz_path=args.data_path, 
        force_preprocess=False
    )
    
    # Extract normalization statistics saved inside the dataset
    label_mean = data.region_dataset.label_mean.to(device)
    label_std = data.region_dataset.label_std.to(device)
    
    # 2. Recreate the EXACT Test Split
    total_regions = len(data.region_dataset)
    train_size = int(0.7 * total_regions)
    val_size = int(0.15 * total_regions)
    test_size = total_regions - train_size - val_size
    
    generator = torch.Generator().manual_seed(args.seed)
    _, _, test_dataset = random_split(
        data.region_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # log all parameters
    logging.info(f"Model Version: {args.version}")
    logging.info(f"Embedding Dimension: {args.embed_dim}")
    logging.info(f"Test Size: {test_size}")
    logging.info(f"Random Seed: {args.seed}")
    
    # We use batch_size=1 or a small batch for clear logging.infoing
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)
    logging.info(f"Test set loaded with {test_size} regions.")
    
    # 3. Load Phase 1 Pre-trained Embeddings
    logging.info("\nLoading Phase 1 Grid Embeddings (E)...")
    final_E = torch.load(f'{weights_dir}/phase1_embeddings_E.pt', map_location=device, weights_only=True)
    final_E = final_E.to(device)
    gn_edge_index = data.gn_edge_index.to(device)
    
    # Generate Region Matrix H
    ada_region_gen = AdaRegionGen()
    H = ada_region_gen.forward(
        regions_gdf=data.regions_gdf, 
        cells_gdf=data.cells_gdf, 
        cell_embeddings=final_E
    ).to(device)
    
    # 4. Load Phase 3 Prompt Enhancer
    logging.info("Loading Phase 3 Prompt Enhancer weights...")
    task_model = DownstreamTaskModel(args).to(device)
    
    # Load the best weights
    task_model.load_state_dict(torch.load(f'{weights_dir}/best_task_model.pth', map_location=device, weights_only=True))
    task_model.eval()
    
    # 5. Inference Loop
    logging.info("\n" + "="*50)
    logging.info(f"{'Region ID':<12} | {'Real Value':<12} | {'Predicted':<12} | {'Abs Error':<12}")
    logging.info("="*50)
    
    all_real = []
    all_pred = []
    all_grid_ids = []
    all_errs = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch_region_idx = batch["region_idx"].to(device)
            sv_images = batch["street_view_images"].to(device)
            raw_tax = batch["raw_tax_sequence"].to(device)
            
            target_grid_indices = batch["grid_indices_in_region"].to(device)
            labels_norm = batch["grid_land_values"].to(device)
            
            batch_H = H[batch_region_idx]
            
            # Forward pass
            predictions_norm = task_model(
                E_grid=final_E, 
                grid_edge_index=gn_edge_index, 
                batch_H=batch_H, 
                sv_images=sv_images, 
                raw_tax_sequences=raw_tax, 
                target_indices=target_grid_indices
            ).view(-1)
            
            # --- UN-NORMALIZE ---
            # Revert from (Mean=0, Std=1) back to the real-world tax rates
            real_values = (labels_norm.view(-1) * label_std) + label_mean
            predicted_values = (predictions_norm * label_std) + label_mean

            real_np = real_values.cpu().numpy()
            pred_np = predicted_values.cpu().numpy()
            grids_np = target_grid_indices.view(-1).cpu().numpy()
            
            all_real.extend(real_np)
            all_pred.extend(pred_np)
            all_grid_ids.extend(grids_np)
            
            sample_size = min(3, len(grids_np)) 
            for i in range(sample_size):
                g_idx = grids_np[i]
                # Look up which region this grid belongs to using the dataset mapping
                r_idx = data.grid_to_region_mapping[g_idx].item()
                r_val = real_np[i]
                p_val = pred_np[i]
                err = abs(r_val - p_val)
                all_errs.append(err)
                logging.info(f"{g_idx:<10} | {r_idx:<10} | {r_val:<10.4f} | {p_val:<10.4f} | {err:<10.4f}")

    # 6. Calculate Final Real-World Metrics
    all_real = np.array(all_real)
    all_pred = np.array(all_pred)
    all_errs = np.array(all_pred - all_real)
    all_errs_total = np.abs(all_real - all_pred)
    
    mae = np.mean(all_errs_total)
    rmse = np.sqrt(np.mean((all_real - all_pred)**2))
    
    logging.info("="*50)
    logging.info(f"FINAL TEST METRICS (Un-Normalized / Real Units)")
    logging.info(f"Mean Absolute Error (MAE): {mae:.4f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    logging.info("="*50)
    
    results_df = pd.DataFrame({
        'node_id': all_grid_ids,
        'real_value': all_real,
        'predicted_value': all_pred,
        'error': all_errs,
        'abs_error': all_errs_total
    })
    
    merged_gdf = data.cells_gdf.merge(results_df, on='node_id', how='inner')
    
    # Save the GeoDataFrame as a CSV
    os.makedirs("./result", exist_ok=True)
    output_path = "./result/test_predictions_spatial.csv"
    merged_gdf.to_csv(output_path, index=False)
    logging.info(f"Spatial predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    parser.add_argument("--embed-dim", type=int, default=144, help="Dimensionality of the embedding space")
    parser.add_argument("--data-path", type=str, default="./dataset/processed_dataset.npz", help="Path to the preprocessed dataset file")
    parser.add_argument("--version", type=str, help="Version of the model to run inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    run_inference(args)