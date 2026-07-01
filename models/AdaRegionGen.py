import torch
import geopandas as gpd

class AdaRegionGen:
    """Adaptive Region Embedding Generation (AdaRegionGen).

    Aggregates fine-grained grid cell embeddings into arbitrary region embeddings
    based on spatial overlap coefficients.
    """

    def __init__(self):
        """Initializes the AdaRegionGen module."""
        pass

    def forward(self, 
                regions_gdf: gpd.GeoDataFrame, 
                cells_gdf: gpd.GeoDataFrame, 
                cell_embeddings: torch.Tensor) -> torch.Tensor:
        """Aggregates cell embeddings into region embeddings using spatial overlap.

        Args:
            regions_gdf (gpd.GeoDataFrame): GeoDataFrame containing the arbitrary region polygons.
            cells_gdf (gpd.GeoDataFrame): GeoDataFrame containing the grid cell polygons.
            cell_embeddings (torch.Tensor): Tensor of shape (m_cells, embed_dim) representing cell embeddings.

        Returns:
            torch.Tensor: Tensor of shape (n_regions, embed_dim) representing the aggregated region embeddings.
        """
        n_regions = len(regions_gdf)
        embed_dim = cell_embeddings.shape[1]
        
        region_embeddings = torch.zeros((n_regions, embed_dim), device=cell_embeddings.device)
        
        if regions_gdf.crs != cells_gdf.crs:
            regions_gdf = regions_gdf.to_crs(cells_gdf.crs)

        cell_areas = cells_gdf.geometry.area.values

        for j, region_row in regions_gdf.iterrows():
            region_geom = region_row.geometry
            
            intersecting_idx = cells_gdf.sindex.query(region_geom, predicate="intersects")
            intersecting_cells = cells_gdf.iloc[intersecting_idx]
            
            if intersecting_cells.empty:
                continue
                
            region_embed = torch.zeros(embed_dim, device=cell_embeddings.device)
            total_overlap = 0.0
            
            for i, cell_row in intersecting_cells.iterrows():
                cell_geom = cell_row.geometry
                
                intersection_area = region_geom.intersection(cell_geom).area
                current_cell_area = cell_areas[i]
                
                if current_cell_area <= 1e-6:
                    overlap_coeff = 0.0
                else:
                    overlap_coeff = intersection_area / current_cell_area
                
                total_overlap += overlap_coeff
                region_embed += overlap_coeff * cell_embeddings[i]
                
            if total_overlap > 1e-6:
                region_embed = region_embed / total_overlap
                
            region_embeddings[j] = region_embed
            
        return region_embeddings