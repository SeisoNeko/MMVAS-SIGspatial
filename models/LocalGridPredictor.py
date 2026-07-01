import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LocalGridPredictor(nn.Module):
    """
    Stage 4 & 5 Module:
    Performs Local-Aware Cross-Attention between grids and regions, followed by 
    residual fusion and final land value prediction.
    """
    def __init__(self, d=144):
        super().__init__()
        
        # 1. The FP32 Shield: Pre-normalizes raw E_grid before GATConv
        self.input_norm = nn.LayerNorm(d)
        
        # 2. Local Aggregation
        self.local_agg = GATConv(d, d, add_self_loops=True)
        
        # 3. Dual LayerNorms for stable feature concatenation
        self.grid_norm = nn.LayerNorm(d)
        self.region_norm = nn.LayerNorm(d)
        
        # 4. Residual MLP Predictor
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, E_grid, grid_edge_index, H_region, target_indices, grid_to_batch_idx):
        """
        Args:
            E_grid: Grid embeddings from Stage 1/2 (m_cells, d)
            grid_edge_index: Graph edges connecting neighboring grids
            H_region: Enriched region embeddings from Stage 3 (n_regions, d)
            target_indices: The specific grid nodes in the current batch
            grid_to_batch_idx: Maps each target grid to its parent region index in H_region
        """
        E_grid_norm = self.input_norm(E_grid)
        # 1. Local Aggregation
        # Shape: (m_cells, d)
        X_grid_local = self.local_agg(E_grid_norm, grid_edge_index) 
        X_grid_batch = X_grid_local[target_indices]
        
        # 2. Normalize local features
        X_grid_batch = self.grid_norm(X_grid_batch)
        
        # 3. Hard Spatial Mapping (The Pivot)
        # Expand region features directly to their child grids using the index map
        # Shape: (num_target_grids, d)
        parent_region_feat = H_region[grid_to_batch_idx]
        parent_region_feat = self.region_norm(parent_region_feat)
        
        # 4. Feature Concatenation
        # Staple the features together. Shape: (num_target_grids, d * 2)
        Z = torch.cat([X_grid_batch, parent_region_feat], dim=-1)
        
        # 5. Final Prediction (Grid Residual)
        grid_residual = self.mlp(Z)
        
        return grid_residual.squeeze(-1) # Shape: (num_target_grids,)