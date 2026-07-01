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
from dataset.mmurgdataset import MMURGDataContainer, collate_fn 

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

def run_citywide_inference(args):
    weights_dir = f"./weights/{args.version}" if args.version else "./weights"
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Running Citywide Deployment (MMURG-Net) on device: {device}")
    
    # 1. Load the Dataset
    print("\nLoading MMURG Data Container...")
    data = MMURGDataContainer(
        config= config
    )
    
    # Extract normalization statistics
    label_mean = data.region_dataset.label_mean.to(device)
    label_std = data.region_dataset.label_std.to(device)
    
    # 2. LOAD THE ENTIRE CITY
    total_regions = len(data.region_dataset)
    # Use the custom collate_fn to properly bundle the grid-level mappings
    full_loader = DataLoader(
        data.region_dataset, batch_size=8, shuffle=False, 
        num_workers=4, collate_fn=collate_fn
    )
    print(f"Citywide dataset loaded with all {total_regions} regions.")
    
    # 3. Load Phase 1 Pre-trained Embeddings & Graph Edges
    print("\nLoading Phase 1 Grid Embeddings (E) and Topology...")
    final_E = torch.load(f'{weights_dir}/phase1_embeddings_E.pt', map_location=device, weights_only=True)
    gn_edge_index = data.gn_edge_index.to(device)
    
    # Generate Region Matrix H
    ada_region_gen = AdaRegionGen()
    H = ada_region_gen.forward(
        regions_gdf=data.regions_gdf, 
        cells_gdf=data.cells_gdf, 
        cell_embeddings=final_E
    ).to(device)
    
    # 4. Load Phase 3 MMURG-Net
    print("Loading Phase 3 MMURG-Net weights...")
    task_model = DownstreamTaskModel(embed_dim=config["model"]["embed_dim"]).to(device)
    task_model.load_state_dict(torch.load(f'{weights_dir}/best_task_model.pth', map_location=device, weights_only=True))
    task_model.eval()
    
    # 5. Inference Loop (Grid-Level)
    print("\nGenerating fine-grained grid predictions for all regions...")
    
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
            
            # Grid-level targets and mappings
            target_grid_indices = batch["grid_indices_in_region"].to(device)
            labels_norm = batch["grid_land_values"].to(device)
            grid_to_batch_idx = batch["grid_to_batch_idx"].to(device)
            
            batch_H = H[region_ids].to(device) 
            
            # Forward pass through MMURG-Net
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
            
            # Un-normalize mathematical components
            # 1. Real Values
            real_values = (labels_norm.view(-1) * label_std) + label_mean
            
            # 2. The Macro Village Baseline
            expanded_baseline_norm = region_baseline[grid_to_batch_idx]
            macro_baseline_real = (expanded_baseline_norm * label_std) + label_mean
            
            # 3. The Micro Grid Residual (Only scale by std, do not add mean to a residual!)
            micro_residual_real = (grid_residual * label_std)
            
            # 4. Final Predicted Value
            final_predicted_real = (final_preds * label_std) + label_mean
            
            # Store data
            all_grid_ids.extend(target_grid_indices.cpu().tolist())
            all_real.extend(real_values.cpu().tolist())
            all_macro_baseline.extend(macro_baseline_real.cpu().tolist())
            all_micro_residual.extend(micro_residual_real.cpu().tolist())
            all_final_pred.extend(final_predicted_real.cpu().tolist())

    # 6. Generate and Save Spatial Files
    print(f"\nCompleted {len(all_grid_ids)} grid-level predictions.")
    print("Generating Citywide Spatial Data for QGIS...")
    
    # Create a DataFrame of the results using Grid node_ids
    results_df = pd.DataFrame({
        'node_id': all_grid_ids,
        'real_value': all_real,
        'macro_baseline': all_macro_baseline,
        'micro_residual': all_micro_residual,
        'final_prediction': all_final_pred
    })
    
    # Calculate Error
    results_df['model_vs_official_diff'] = results_df['final_prediction'] - results_df['real_value']
    
    # Merge the results back onto the fine-grained Hexagon geographic polygons
    merged_gdf = data.cells_gdf.merge(results_df, on='node_id', how='inner')

    # combine hex grid locale information
    merged_gdf_web = merged_gdf.to_crs(epsg=3826)
    merged_gdf_web['center_lng'] = merged_gdf_web.geometry.centroid.x
    merged_gdf_web['center_lat'] = merged_gdf_web.geometry.centroid.y
    # Extract Bounding Box (minx=left, miny=bottom, maxx=right, maxy=top)
    bounds = merged_gdf_web.geometry.bounds
    merged_gdf_web['left_lng'] = bounds['minx']    # West bound
    merged_gdf_web['bottom_lat'] = bounds['miny']  # South bound
    merged_gdf_web['right_lng'] = bounds['maxx']   # East bound
    merged_gdf_web['top_lat'] = bounds['maxy']     # North bound
    df_for_web = pd.DataFrame(merged_gdf_web.drop(columns=['geometry']))
    
    # Output Paths
    os.makedirs("./result", exist_ok=True)
    csv_output_path = f"./result/{args.output_name}.csv"
    json_output_path = f"./result/{args.output_name}.json"
    gpkg_output_path = f"./result/{args.output_name}.gpkg"
    
    # Save CSV
    df_for_web.to_csv(csv_output_path, index=False)
    print(f"Successfully saved CSV to: {csv_output_path}")

    # Save json
    df_for_web.to_json(json_output_path, orient="records")
    print(f"Successfully saved json to: {json_output_path}")
    
    # Save GPKG (GeoPackage is heavily preferred over GeoJSON for QGIS performance)
    merged_gdf_web.to_file(gpkg_output_path, driver="GPKG")
    print(f"Successfully saved GeoPackage to: {gpkg_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    parser.add_argument("--embed-dim", type=int, default=144, help="Dimensionality of the embedding space")
    parser.add_argument("--data-path", type=str, default="./dataset/processed_dataset.npz", help="Path to the preprocessed dataset file")
    parser.add_argument("--version", type=str, help="Version of the model to run inference (e.g., v2)")
    parser.add_argument("--output-name", type=str, default="inference_predictions", help="Name of the output files")
    args = parser.parse_args()
    
    run_citywide_inference(args)