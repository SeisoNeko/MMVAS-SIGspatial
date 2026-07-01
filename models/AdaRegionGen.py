import torch
import geopandas as gpd

class AdaRegionGen:
    """
    Adaptive Region Embedding Generation (AdaRegionGen)
    Aggregates fine-grained grid cell embeddings into arbitrary region embeddings 
    based on spatial overlap coefficients.
    """
    def __init__(self):
        # This module requires no neural network parameters or training.
        pass

    def forward(self, 
                regions_gdf: gpd.GeoDataFrame, 
                cells_gdf: gpd.GeoDataFrame, 
                cell_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            regions_gdf: GeoDataFrame containing the arbitrary region polygons (e.g., land use zones).
            cells_gdf: GeoDataFrame containing the hexagonal grid cell polygons. 
                       Must be in the same coordinate reference system (CRS) as regions_gdf.
            cell_embeddings: Tensor of shape (m_cells, embed_dim) representing the output E from GridLearner.
            
        Returns:
            region_embeddings: Tensor of shape (n_regions, embed_dim) representing the final matrix H.
        """
        n_regions = len(regions_gdf)
        embed_dim = cell_embeddings.shape[1]
        
        # Initialize the output matrix H
        region_embeddings = torch.zeros((n_regions, embed_dim), device=cell_embeddings.device)
        
        # It is critical that both geographic dataframes use the same CRS for accurate area math
        if regions_gdf.crs!= cells_gdf.crs:
            regions_gdf = regions_gdf.to_crs(cells_gdf.crs)

        # Pre-calculate the total area for all grid cells (Area(c_i))
        cell_areas = cells_gdf.geometry.area.values

        # Iterate through each target region r_j
        for j, region_row in regions_gdf.iterrows():
            region_geom = region_row.geometry
            
            # Step 1: Identify the subset of grid cells C_{r_j} that intersect with region r_j
            # Using spatial indexing (sindex) under the hood in geopandas makes this efficient
            intersecting_idx = cells_gdf.sindex.query(region_geom, predicate="intersects")
            intersecting_cells = cells_gdf.iloc[intersecting_idx]
            
            if intersecting_cells.empty:
                continue
                
            region_embed = torch.zeros(embed_dim, device=cell_embeddings.device)
            total_overlap = 0.0
            
            # Step 2: Calculate the precise overlapping area coefficient for each intersecting cell
            for i, cell_row in intersecting_cells.iterrows():
                cell_geom = cell_row.geometry
                
                # Area(r_j \cap c_i)
                intersection_area = region_geom.intersection(cell_geom).area
                
                # o_{r_j \cap c_i} = Area(r_j \cap c_i) / Area(c_i)
                current_cell_area = cell_areas[i]
                if current_cell_area <= 1e-6:
                    overlap_coeff = 0.0
                else:
                    overlap_coeff = intersection_area / current_cell_area
                
                total_overlap += overlap_coeff
                # Step 3: Compute coefficient-weighted sum of constituent cell embeddings
                # h_j = \sum (o_{r_j \cap c_i} * e_i)
                region_embed += overlap_coeff * cell_embeddings[i]
                
            if total_overlap > 1e-6:
                region_embed = region_embed / total_overlap
            # Store the aggregated embedding h_j into the final matrix H
            region_embeddings[j] = region_embed
            
        return region_embeddings
