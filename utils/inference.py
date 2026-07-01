import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import sys
import argparse
import yaml

sys.path.append(os.getcwd())

from models.AdaRegionGen import AdaRegionGen
from models.DownStreamTask import DownstreamTaskModel
from dataset.mmvasdataset import MMVASDataContainer, collate_fn 

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file.

    Args:
        config_path (str, optional): The file path to the YAML configuration. 
            Defaults to "config.yaml".

    Returns:
        dict: A dictionary containing the parsed configuration parameters.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

def run_citywide_inference(args):
    """Executes citywide spatial inference using the trained prediction pipeline.

    This function loads the macro-region and micro-grid datasets, applies pre-trained 
    embeddings, and computes the final predicted values by combining baseline regional 
    estimates with localized grid residuals. Outputs are exported to CSV, JSON, and GPKG.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing 
            device preferences, model versioning, and output configurations.
    """
    weights_dir = f"./weights/{args.version}" if args.version else "./weights"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running citywide inference on device: {device}")
    
    print("\nLoading data container...")
    data = MMVASDataContainer(
        config=config
    )
    
    label_mean = data.region_dataset.label_mean.to(device)
    label_std = data.region_dataset.label_std.to(device)
    
    total_regions = len(data.region_dataset)
    
    full_loader = DataLoader(
        data.region_dataset, batch_size=8, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )
    print(f"Dataset loaded with {total_regions} regions.")
    
    print("\nLoading pre-trained grid embeddings and geographic network topology...")
    final_E = torch.load(f'{weights_dir}/phase1_embeddings_E.pt', map_location=device, weights_only=True)
    gn_edge_index = data.gn_edge_index.to(device)
    
    ada_region_gen = AdaRegionGen()
    H = ada_region_gen.forward(
        regions_gdf=data.regions_gdf, 
        cells_gdf=data.cells_gdf, 
        cell_embeddings=final_E
    ).to(device)
    
    print("Loading downstream task model weights...")
    task_model = DownstreamTaskModel(embed_dim=config["model"]["embed_dim"]).to(device)
    task_model.load_state_dict(torch.load(f'{weights_dir}/best_task_model.pth', map_location=device, weights_only=True))
    task_model.eval()
    
    print("\nGenerating fine-grained grid predictions...")
    
    all_grid_ids = []
    all_real = []
    all_final_pred = []
    all_macro_baseline = []
    all_micro_residual = []
    
    with torch.no_grad():
        for batch in full_loader:
            region_ids = batch["region_idx"]
            raw_tax_sequences = batch["raw_tax_sequence"].to(device)
            sv_images = batch["street_view_images"].to(device)
            
            target_grid_indices = batch["grid_indices_in_region"].to(device)
            labels_norm = batch["grid_land_values"].to(device)
            grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
            
            batch_H = H[region_ids].to(device) 
            
            final_preds, region_baseline = task_model(
                E_grid=final_E, 
                grid_edge_index=gn_edge_index, 
                batch_H=batch_H, 
                sv_images=sv_images, 
                raw_tax_sequences=raw_tax_sequences, 
                target_indices=target_grid_indices,
                grid_to_batch_idx=grid_to_batch_idx
            )
            grid_residual = final_preds - region_baseline[grid_to_batch_idx]
            
            real_values = (labels_norm.view(-1) * label_std) + label_mean
            
            expanded_baseline_norm = region_baseline[grid_to_batch_idx]
            macro_baseline_real = (expanded_baseline_norm * label_std) + label_mean
            
            micro_residual_real = (grid_residual * label_std)
            
            final_predicted_real = (final_preds * label_std) + label_mean
            
            all_grid_ids.extend(target_grid_indices.cpu().tolist())
            all_real.extend(real_values.cpu().tolist())
            all_macro_baseline.extend(macro_baseline_real.cpu().tolist())
            all_micro_residual.extend(micro_residual_real.cpu().tolist())
            all_final_pred.extend(final_predicted_real.cpu().tolist())

    print(f"\nCompleted {len(all_grid_ids)} grid-level predictions.")
    print("Generating spatial output files...")
    
    results_df = pd.DataFrame({
        'node_id': all_grid_ids,
        'real_value': all_real,
        'macro_baseline': all_macro_baseline,
        'micro_residual': all_micro_residual,
        'final_prediction': all_final_pred
    })
    
    results_df['model_vs_official_diff'] = results_df['final_prediction'] - results_df['real_value']
    
    merged_gdf = data.cells_gdf.merge(results_df, on='node_id', how='inner')

    merged_gdf_web = merged_gdf.to_crs(epsg=3826)
    merged_gdf_web['center_lng'] = merged_gdf_web.geometry.centroid.x
    merged_gdf_web['center_lat'] = merged_gdf_web.geometry.centroid.y
    
    bounds = merged_gdf_web.geometry.bounds
    merged_gdf_web['left_lng'] = bounds['minx']    
    merged_gdf_web['bottom_lat'] = bounds['miny']  
    merged_gdf_web['right_lng'] = bounds['maxx']   
    merged_gdf_web['top_lat'] = bounds['maxy']     
    df_for_web = pd.DataFrame(merged_gdf_web.drop(columns=['geometry']))
    
    os.makedirs("./result", exist_ok=True)
    csv_output_path = f"./result/{args.output_name}.csv"
    json_output_path = f"./result/{args.output_name}.json"
    gpkg_output_path = f"./result/{args.output_name}.gpkg"
    
    df_for_web.to_csv(csv_output_path, index=False)
    print(f"Saved CSV output to: {csv_output_path}")

    df_for_web.to_json(json_output_path, orient="records")
    print(f"Saved JSON output to: {json_output_path}")
    
    merged_gdf_web.to_file(gpkg_output_path, driver="GPKG")
    print(f"Saved GeoPackage output to: {gpkg_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to execute inference on (cuda or cpu)")
    parser.add_argument("--embed-dim", type=int, default=144, help="Dimensionality of the embedding space")
    parser.add_argument("--data-path", type=str, default="./dataset/processed_dataset.npz", help="Path to the preprocessed dataset file")
    parser.add_argument("--version", type=str, help="Version identifier of the model weights to load")
    parser.add_argument("--output-name", type=str, default="inference_predictions", help="Base filename for the output files")
    args = parser.parse_args()
    
    run_citywide_inference(args)