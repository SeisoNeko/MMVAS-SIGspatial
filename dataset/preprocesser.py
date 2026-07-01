import torch
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from libpysal.weights import KNN
import gc
import warnings
from tqdm import tqdm

DATA_DIR = Path("./data/new")
TARGET_CRS = 3826

def load_land_use_mapping():
    """Loads the land use RGB-to-category mapping.

    Returns:
        dict: A dictionary mapping RGB string codes to land use categories.
    """
    mapping_df = pd.read_csv(DATA_DIR / "landUse" / "land_use_map.csv")
    map_dict = {}
    for _, row in mapping_df.iterrows():
        rgb_code = f"{row['R']}{row['G']}{row['B']}"
        map_dict[rgb_code] = row['category']
    return map_dict

def load_and_align(filepath, layer=None):
    """Loads a spatial dataset and aligns it to the target Coordinate Reference System.

    Args:
        filepath (str or Path): The path to the spatial data file.
        layer (str, optional): The specific layer to load for multi-layer files. Defaults to None.

    Returns:
        gpd.GeoDataFrame: The loaded and aligned spatial dataframe.
    """
    if layer:
        gdf = gpd.read_file(filepath, layer=layer)
    else:
        gdf = gpd.read_file(filepath)

    if gdf.crs is None or gdf.crs.to_epsg() != TARGET_CRS:
        gdf = gdf.to_crs(epsg=TARGET_CRS)
    return gdf

def build_feature_knn_graph(feature_matrix, k=6):
    """Builds a k-Nearest Neighbors graph based on feature cosine similarity.

    Args:
        feature_matrix (np.ndarray): The input feature matrix of shape (m, features).
        k (int, optional): The number of nearest neighbors to compute. Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - edge_index (np.ndarray): The edge indices of shape (2, num_edges).
            - edge_weights (np.ndarray): The computed cosine similarity weights.
    """
    m = feature_matrix.shape[0]
    
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
    nn.fit(feature_matrix)
    
    distances, indices = nn.kneighbors(feature_matrix)
    
    source_nodes = []
    target_nodes = []
    edge_weights = []
    
    for i in range(m):
        for j in range(1, k+1):
            neighbor_idx = indices[i, j]
            similarity = 1.0 - distances[i, j] 
            
            source_nodes.append(i)
            target_nodes.append(neighbor_idx)
            edge_weights.append(similarity)
            
    edge_index = np.array([source_nodes, target_nodes])
    edge_weights = np.array(edge_weights, dtype=np.float32)
    return edge_index, edge_weights

def create_grid_region_mapping(cells_gdf, regions_gdf):
    """Creates a mapping array linking grid cells to their enclosing geographic regions.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        regions_gdf (gpd.GeoDataFrame): The target macroscopic regions.

    Returns:
        np.ndarray: An array mapping grid node IDs to region indices.
    """
    print("\n--- Creating Grid-to-Region Mapping ---")
    centroids = cells_gdf.copy()
    centroids['geometry'] = centroids.geometry.centroid
    
    joined = gpd.sjoin(centroids[['node_id', 'geometry']], 
                       regions_gdf[['region_idx', 'geometry']], 
                       how='left', predicate='intersects')
    joined = joined.drop_duplicates(subset=['node_id'])
    
    missing_mask = joined['region_idx'].isna()
    if missing_mask.any():
        missing_nodes = joined.loc[missing_mask, 'node_id']
        missing_geom = centroids.loc[missing_nodes, 'geometry']
        missing_gdf = gpd.GeoDataFrame({'node_id': missing_nodes, 'geometry': missing_geom}, crs=TARGET_CRS)
        
        nearest = gpd.sjoin_nearest(missing_gdf, regions_gdf[['region_idx', 'geometry']], how='left')
        nearest = nearest.drop_duplicates(subset=['node_id'])
        
        joined.set_index('node_id', inplace=True)
        nearest.set_index('node_id', inplace=True)
        joined.update(nearest)
        joined.reset_index(inplace=True)
        
    mapping = joined.sort_values('node_id')['region_idx'].astype(int).to_numpy()
    return mapping

def process_farbac_pipeline(cells_gdf, m, k=6):
    """Processes Floor Area Ratio and Building Coverage Ratio feature structures.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        m (int): The total number of grid cells.
        k (int, optional): Number of neighbors for graph construction. Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - farbac_feat (np.ndarray): The extracted feature matrix.
            - farbac_edge_index (np.ndarray): Edge indices for the feature graph.
            - farbac_edge_weights (np.ndarray): Edge weights for the feature graph.
    """
    print("\n--- Processing FAR/BAC ---")
    farbac_raw = load_and_align(DATA_DIR / "FARBAC" / "build_rate.gpkg")
    
    for col in ['BUILDRATE', 'VOLUMERATE']:
        farbac_raw[col] = (farbac_raw[col].astype(str)
                           .str.replace('%', '', regex=False)
                           .str.replace('*', '', regex=False)
                           .str.replace(r'[^\d\.\-]', '', regex=True)
                           .replace(['NULL', 'nan', ''], '0')
                           .astype(float))

    farbac_raw['centroid'] = farbac_raw.geometry.centroid
    farbac_points = farbac_raw.set_geometry('centroid')
    
    joined_farbac = gpd.sjoin(farbac_points, cells_gdf[['node_id', 'geometry']], how='inner', predicate='intersects')
    farbac_agg = joined_farbac.groupby('node_id')[['VOLUMERATE', 'BUILDRATE']].mean().fillna(0)
    
    farbac_feat_df = pd.DataFrame(index=range(m)).join(farbac_agg).fillna(0)
    farbac_feat = farbac_feat_df[['VOLUMERATE', 'BUILDRATE']].to_numpy(dtype=np.float32)
    
    farbac_edge_index, farbac_edge_weights = build_feature_knn_graph(farbac_feat, k=k)

    del farbac_raw, farbac_points, joined_farbac, farbac_agg, farbac_feat_df
    gc.collect()
    
    return farbac_feat, farbac_edge_index, farbac_edge_weights

def process_lu_pipeline(cells_gdf, m, k=6):
    """Processes land use intersection and area aggregation to extract tabular features.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        m (int): The total number of grid cells.
        k (int, optional): Number of neighbors for graph construction. Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - lu_feat (np.ndarray): The normalized land use feature matrix.
            - lu_edge_index (np.ndarray): Edge indices for the feature graph.
            - lu_edge_weights (np.ndarray): Edge weights for the feature graph.
    """
    print("\n--- Processing Land Use ---")
    lu_raw = load_and_align(DATA_DIR / "landUse" / "land_use_res_1.gpkg")
    LU_MAPPING = load_land_use_mapping()
    
    lu_raw['DN'] = lu_raw['DN'].astype(str).str.zfill(9)
    lu_raw['lu_category'] = lu_raw['DN'].map(LU_MAPPING).fillna('Unknown')
    
    chunk_size = 500
    tabular_results = []
    
    for i in tqdm(range(0, m, chunk_size), desc="Intersecting LU Chunks"):
        cells_chunk = cells_gdf.iloc[i : i + chunk_size][['node_id', 'geometry']]
        
        relevant_lu = gpd.sjoin(lu_raw[['lu_category', 'geometry']], cells_chunk, how='inner', predicate='intersects')
        relevant_lu = relevant_lu.drop_duplicates(subset=['geometry']).drop(columns=['index_right', 'node_id'])
        
        if relevant_lu.empty:
            continue

        relevant_lu['geometry'] = relevant_lu['geometry'].simplify(tolerance=0.5, preserve_topology=True)
        
        intersected_chunk = gpd.overlay(cells_chunk, relevant_lu, how='intersection')
        if intersected_chunk.empty:
            continue

        intersected_chunk['area'] = intersected_chunk.geometry.area
        chunk_agg = intersected_chunk.groupby(['node_id', 'lu_category'])['area'].sum().reset_index()
        tabular_results.append(chunk_agg)

        del cells_chunk, relevant_lu, intersected_chunk, chunk_agg
        gc.collect()

    if tabular_results:
        final_tabular = pd.concat(tabular_results, ignore_index=True)
        lu_pivot = final_tabular.pivot_table(index='node_id', columns='lu_category', values='area', aggfunc='sum', fill_value=0)
    else:
        lu_pivot = pd.DataFrame(index=range(m))

    hex_areas = cells_gdf.set_index('node_id').geometry.area
    lu_pivot = lu_pivot.div(hex_areas, axis=0).fillna(0)

    lu_feat_df = pd.DataFrame(index=range(m)).join(lu_pivot).fillna(0)
    lu_feat = lu_feat_df.to_numpy(dtype=np.float32)

    lu_edge_index, lu_edge_weights = build_feature_knn_graph(lu_feat, k=k)

    del lu_raw, final_tabular, lu_pivot, lu_feat_df
    gc.collect()
    
    return lu_feat, lu_edge_index, lu_edge_weights

def process_poi_pipeline(cells_gdf, m, min_occurrences=10, k=6):
    """Aggregates Points of Interest (POI) data into grid-level features.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        m (int): The total number of grid cells.
        min_occurrences (int, optional): Minimum frequency required to retain a POI category. Defaults to 10.
        k (int, optional): Number of neighbors for graph construction. Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - poi_feat (np.ndarray): The POI frequency feature matrix.
            - poi_edge_index (np.ndarray): Edge indices for the feature graph.
            - poi_edge_weights (np.ndarray): Edge weights for the feature graph.
            - poi_gdf (gpd.GeoDataFrame): The aligned POI geodataframe.
    """
    print("\n--- Processing POI ---")
    poi_df = pd.read_csv(DATA_DIR / "POIs" / "tainan_pois.csv")
    
    poi_df['category'] = poi_df['category'].replace(r'^\s*$', np.nan, regex=True)
    poi_df['unified_type'] = poi_df['category'].fillna(poi_df['poi_type']).astype(str)
    poi_df['lng'] = pd.to_numeric(poi_df['lng'], errors='coerce')
    poi_df['lat'] = pd.to_numeric(poi_df['lat'], errors='coerce')

    poi_gdf = gpd.GeoDataFrame(poi_df, geometry=gpd.points_from_xy(poi_df['lng'], poi_df['lat']), crs="EPSG:4326").to_crs(epsg=3826)
    joined_poi_grid = gpd.sjoin(poi_gdf, cells_gdf[['node_id', 'geometry']], how='inner', predicate='intersects')

    poi_counts = pd.crosstab(joined_poi_grid['node_id'], joined_poi_grid['unified_type'])
    category_totals = poi_counts.sum()
    valid_types = category_totals[category_totals >= min_occurrences].index

    poi_filtered = poi_counts[valid_types]
    poi_feat_df = pd.DataFrame(index=range(m)).join(poi_filtered).fillna(0)
    poi_feat = poi_feat_df.to_numpy(dtype=np.float32)
    
    poi_edge_index, poi_edge_weights = build_feature_knn_graph(poi_feat, k=k)

    del poi_df, poi_counts, poi_filtered, poi_feat_df
    gc.collect()

    return poi_feat, poi_edge_index, poi_edge_weights, poi_gdf

def process_gn_pipeline(cells_gdf, m, k=6):
    """Computes the physical geographic adjacency graph.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        m (int): The total number of grid cells.
        k (int, optional): Number of geographic nearest neighbors. Defaults to 6.

    Returns:
        tuple: A tuple containing:
            - gn_feat (np.ndarray): Matrix containing distance values to neighbors.
            - gn_edge_index (np.ndarray): Edge indices for the geographic network.
    """
    print(f"\n--- Processing Physical Neighbors (GN) (k={k}) ---")
    centroids = cells_gdf.geometry.centroid
    w_knn = KNN.from_dataframe(cells_gdf, k=k)
    
    src, dst = [], []
    for node, neighbors in w_knn.neighbors.items():
        for neighbor in neighbors:
            src.append(node)
            dst.append(neighbor)
            
    gn_edge_index = np.array([src, dst], dtype=np.int64)
    gn_feat = np.zeros((m, k), dtype=np.float32)
    centroids = cells_gdf.geometry.centroid
    
    for i in range(m):
        neighbors = w_knn.neighbors.get(i, [])
        dists = [centroids.iloc[i].distance(centroids.iloc[n]) for n in neighbors]
        num_neighbors = min(len(dists), k)
        gn_feat[i, :num_neighbors] = dists[:num_neighbors]

    return gn_feat, gn_edge_index

def generate_region_text_prompts(regions_gdf, poi_gdf):
    """Generates natural language prompts representing urban environments at a macroscopic scale.

    Args:
        regions_gdf (gpd.GeoDataFrame): The macro-level geographic regions.
        poi_gdf (gpd.GeoDataFrame): The points of interest.

    Returns:
        list: A list of formatted string prompts.
    """
    print("\n--- Generating Region-Level Text Prompts ---")
    joined_poi_region = gpd.sjoin(poi_gdf, regions_gdf[['region_idx', 'geometry']], how='inner', predicate='intersects')
    poi_counts = pd.crosstab(joined_poi_region['region_idx'], joined_poi_region['unified_type']) if not joined_poi_region.empty else pd.DataFrame()
    centroids_4326 = regions_gdf.geometry.centroid.to_crs(epsg=4326)

    raw_prompts = []
    for i, row in regions_gdf.iterrows():
        r_idx = row['region_idx']
        lat, lng = centroids_4326.iloc[i].y, centroids_4326.iloc[i].x
        
        village_name = row.get('VILLNAME', f'Region {r_idx}')
        town_name = row.get('TOWNNAME', '')
        county = row.get('COUNTYNAME', 'Tainan City')
        address = ", ".join([p for p in [village_name, town_name, county, "Taiwan"] if p])
        
        poi_text_parts = []
        total_pois = 0
        if r_idx in poi_counts.index:
            region_pois = poi_counts.loc[r_idx]
            total_pois = region_pois.sum()
            if total_pois > 0:
                sorted_pois = region_pois[region_pois > 0].sort_values(ascending=False)
                for category, count in sorted_pois.items():
                    poi_text_parts.append(f"{int(count)} {category} ({round((count / total_pois) * 100, 2)}%)")
                        
        poi_string = ", ".join(poi_text_parts) if total_pois > 0 else "0 POIs"
        prompt = (f"Please infer the urban environment of this region:\n"
                  f"The region represents {address}.\n\n"
                  f"Centroid Coordinates: ({lat:.5f}, {lng:.5f})\n\n"
                  f"Included POIs: The region contains a total of {int(total_pois)} POIs. "
                  f"The distribution is: {poi_string}.")
        raw_prompts.append(prompt)
        
    return raw_prompts

def generate_grid_text_prompts(cells_gdf, regions_gdf, poi_gdf):
    """Generates natural language prompts representing urban environments at a microscopic grid scale.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        regions_gdf (gpd.GeoDataFrame): The encompassing regions for addressing context.
        poi_gdf (gpd.GeoDataFrame): The points of interest.

    Returns:
        list: A list of formatted string prompts representing individual grids.
    """
    print("\n--- Generating Grid-Level Text Prompts ---")
    
    joined_poi_grid = gpd.sjoin(poi_gdf, cells_gdf[['node_id', 'geometry']], how='inner', predicate='intersects')
    poi_counts = pd.crosstab(joined_poi_grid['node_id'], joined_poi_grid['unified_type']) if not joined_poi_grid.empty else pd.DataFrame()
    
    centroids = cells_gdf.copy()
    centroids['geometry'] = centroids.geometry.centroid
    joined_grid_region = gpd.sjoin(
        centroids[['node_id', 'geometry']], 
        regions_gdf[['region_idx', 'VILLNAME', 'TOWNNAME', 'COUNTYNAME', 'geometry']], 
        how='left', 
        predicate='intersects'
    )
    joined_grid_region = joined_grid_region.drop_duplicates(subset=['node_id']).set_index('node_id')
    
    centroids_4326 = cells_gdf.geometry.centroid.to_crs(epsg=4326)

    raw_prompts = []
    for i, row in tqdm(cells_gdf.iterrows(), total=len(cells_gdf), desc="Generating Grid Prompts"):
        node_id = row['node_id']
        lat, lng = centroids_4326.iloc[i].y, centroids_4326.iloc[i].x
        
        if node_id in joined_grid_region.index:
            r_row = joined_grid_region.loc[node_id]
            address = ", ".join([str(p) for p in [r_row.get('VILLNAME', ''), r_row.get('TOWNNAME', ''), r_row.get('COUNTYNAME', 'Tainan City'), "Taiwan"] if pd.notna(p) and p != ''])
        else:
            address = "Tainan City, Taiwan"
            
        poi_text_parts = []
        total_pois = 0
        if node_id in poi_counts.index:
            grid_pois = poi_counts.loc[node_id]
            total_pois = grid_pois.sum()
            if total_pois > 0:
                sorted_pois = grid_pois[grid_pois > 0].sort_values(ascending=False)
                for category, count in sorted_pois.items():
                    poi_text_parts.append(f"{int(count)} {category}")
                    
        poi_string = ", ".join(poi_text_parts) if total_pois > 0 else "0 POIs"
        
        prompt = (f"Please infer the urban environment of this 100m micro-grid:\n"
                  f"The grid is located in {address}.\n\n"
                  f"Centroid Coordinates: ({lat:.5f}, {lng:.5f})\n\n"
                  f"Included POIs: The grid contains a total of {int(total_pois)} POIs. "
                  f"The distribution is: {poi_string}.")
        raw_prompts.append(prompt)
        
    return raw_prompts

def process_tax_pipeline_100m_imputed(cells_gdf, regions_gdf, grid_mapping, seq_len=3, raw_tax_dim=3):
    """Extracts historical sequences and spatial grid labels from imputed assessment data.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids.
        regions_gdf (gpd.GeoDataFrame): The macro-level geographic regions.
        grid_mapping (np.ndarray): The mapping array linking grids to regions.
        seq_len (int, optional): The number of time steps in the history sequence. Defaults to 3.
        raw_tax_dim (int, optional): The dimensionality of the historical features. Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - tax_sequences (torch.Tensor): The temporal sequences per region.
            - grid_labels (torch.Tensor): The regression target variables for each grid.
    """
    print("\n--- Processing Imputed Tax Data for 100m Hex Grid ---")
    imputed_tax_csv_path = DATA_DIR / "tax" / "100m_hex_tax_imputed.csv"
    tax_df = pd.read_csv(imputed_tax_csv_path)

    cells_with_tax = cells_gdf[['node_id', 'id']].merge(tax_df, on='id', how='left')
    cells_with_tax = cells_with_tax.sort_values('node_id').reset_index(drop=True)

    for col in tax_df.select_dtypes(include=[np.number]).columns:
        if col in cells_with_tax.columns:
            cells_with_tax[col] = cells_with_tax[col].fillna(cells_with_tax[col].median())

    grid_labels = cells_with_tax['avg_111_land_rate'].values
    grid_labels = torch.tensor(grid_labels, dtype=torch.float32)

    cells_with_tax['region_id'] = grid_mapping
    region_agg = cells_with_tax.groupby('region_id').mean(numeric_only=True)

    num_regions = len(regions_gdf)

    tax_sequences = torch.zeros((num_regions, seq_len, raw_tax_dim), dtype=torch.float32)
    city_mean_row = cells_with_tax.mean(numeric_only=True)

    for r_idx in range(num_regions):
        if r_idx in region_agg.index:
            row = region_agg.loc[r_idx]
        else:
            row = city_mean_row 

        tax_sequences[r_idx, 0, 0] = row['avg_105_land_rate']
        tax_sequences[r_idx, 0, 1] = row['avg_105_present_value']
        tax_sequences[r_idx, 0, 2] = row['weighted_volumerate_avg']
        
        tax_sequences[r_idx, 1, 0] = row['avg_108_land_rate']
        tax_sequences[r_idx, 1, 1] = row['avg_108_present_value']
        tax_sequences[r_idx, 1, 2] = row['weighted_volumerate_avg']
        
        tax_sequences[r_idx, 2, 0] = row['avg_111_land_rate']
        tax_sequences[r_idx, 2, 1] = row['avg_111_present_value']
        tax_sequences[r_idx, 2, 2] = row['weighted_volumerate_avg']

    return tax_sequences, grid_labels

def export_GPKG(cells_gdf, grid_labels, grid_to_region_mapping, farbac_feat):
    """Exports processed features alongside geographic primitives for external verification.

    Args:
        cells_gdf (gpd.GeoDataFrame): The spatial grids geometries.
        grid_labels (torch.Tensor): Processed labels corresponding to grids.
        grid_to_region_mapping (np.ndarray): The mapping between grids and regions.
        farbac_feat (np.ndarray): The extracted FAR/BAC features.
    """
    print("\n--- Exporting Verification Files ---")
    
    verify_gdf = cells_gdf[['node_id', 'id', 'geometry']].copy()
    
    verify_gdf['target_111_rate'] = grid_labels.numpy()
    verify_gdf['region_idx'] = grid_to_region_mapping
    verify_gdf['farbac_volume_rate'] = farbac_feat[:, 0]
    
    OUTPUT_DIR = Path("./dataset")
    csv_out_path = OUTPUT_DIR / "qgis_verify_dataset.csv"
    verify_gdf.to_csv(csv_out_path, index=False)
    
    gpkg_out_path = OUTPUT_DIR / "qgis_verify_dataset.gpkg"
    verify_gdf.to_file(gpkg_out_path, driver="GPKG")

def main(grid_type: str = "100m", knn_k: int = 6, poi_min_occurrences: int = 10, tax_seq_len: int = 3, tax_raw_dim: int = 3):
    """Main execution pipeline for generating all spatial graphs and aligned feature matrices.

    Args:
        grid_type (str, optional): The resolution scale of the base grid. Defaults to "100m".
        knn_k (int, optional): K-value used for generating all nearest neighbor graphs. Defaults to 6.
        poi_min_occurrences (int, optional): Filter threshold for POI categories. Defaults to 10.
        tax_seq_len (int, optional): Expected sequence length for historical data. Defaults to 3.
        tax_raw_dim (int, optional): Number of distinct historical features per time step. Defaults to 3.

    Returns:
        dict: A compiled dictionary of all normalized tensors, mappings, and edge indices.
    """
    if grid_type == "1km":
        cells_gdf = load_and_align(DATA_DIR / "grid" / "grid.gpkg", layer="grid")
    elif grid_type == "100m":
        cells_gdf = load_and_align(DATA_DIR / "grid" / "grid_100m_hex.gpkg", layer="clipped")
    regions_gdf = load_and_align(DATA_DIR / "region" / "tainan_village_3826.gpkg")

    if 'id' in cells_gdf.columns:
        cells_gdf = cells_gdf.sort_values('id').reset_index(drop=True)
    else:
        cells_gdf = cells_gdf.reset_index(drop=True)

    cells_gdf['node_id'] = cells_gdf.index 
    regions_gdf['region_idx'] = range(len(regions_gdf))
    
    m = len(cells_gdf)

    grid_to_region_mapping = create_grid_region_mapping(cells_gdf, regions_gdf)

    poi_feat, poi_edge_index, poi_edge_weights, poi_gdf = process_poi_pipeline(cells_gdf, m, min_occurrences=poi_min_occurrences, k=knn_k)

    raw_text_prompts = generate_grid_text_prompts(cells_gdf, regions_gdf, poi_gdf)

    raw_tax_sequences, grid_labels = process_tax_pipeline_100m_imputed(cells_gdf, regions_gdf, grid_to_region_mapping, seq_len=tax_seq_len, raw_tax_dim=tax_raw_dim)

    farbac_feat, farbac_edge_index, farbac_edge_weights = process_farbac_pipeline(cells_gdf, m, k=knn_k)
    
    lu_feat, lu_edge_index, lu_edge_weights = process_lu_pipeline(cells_gdf, m, k=knn_k)
    
    gn_feat, gn_edge_index = process_gn_pipeline(cells_gdf, m, k=knn_k)
    
    dataset = {
        "grid_to_region_mapping": grid_to_region_mapping,
        "grid_labels": grid_labels,
        "farbac_feat": farbac_feat, "farbac_edge_index": farbac_edge_index, "farbac_edge_weights": farbac_edge_weights,
        "lu_feat": lu_feat, "lu_edge_index": lu_edge_index, "lu_edge_weights": lu_edge_weights,
        "poi_feat": poi_feat, "poi_edge_index": poi_edge_index, "poi_edge_weights": poi_edge_weights,
        "gn_feat": gn_feat, "gn_edge_index": gn_edge_index,
        "raw_text_prompts": raw_text_prompts,
        "raw_tax_sequences": raw_tax_sequences
    }

    assert all(x == m for x in [
        grid_labels.shape[0], grid_to_region_mapping.shape[0], 
        poi_feat.shape[0], lu_feat.shape[0], farbac_feat.shape[0], gn_feat.shape[0]
    ])

    export_GPKG(cells_gdf, grid_labels, grid_to_region_mapping, farbac_feat)

    return dataset

if __name__ == "__main__":
    model_dataset = main()
    OUTPUT_DIR = Path("./dataset")
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(OUTPUT_DIR / "processed_dataset.npz", **model_dataset)