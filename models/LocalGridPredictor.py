import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LocalGridPredictor(nn.Module):
    """Performs local-aware feature aggregation and residual fusion for final grid prediction.

    This module integrates microscopic grid embeddings with macroscopic region
    embeddings to predict the final target values via a residual multilayer perceptron.
    """

    def __init__(self, d=144):
        """Initializes the LocalGridPredictor.

        Args:
            d (int, optional): The dimensionality of the input node and region embeddings. Defaults to 144.
        """
        super().__init__()
        
        self.input_norm = nn.LayerNorm(d)
        
        self.local_agg = GATConv(d, d, add_self_loops=True)
        
        self.grid_norm = nn.LayerNorm(d)
        self.region_norm = nn.LayerNorm(d)
        
        self.mlp = nn.Sequential(
            nn.Linear(d * 2, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, E_grid, grid_edge_index, H_region, target_indices, grid_to_batch_idx):
        """Executes the forward pass for grid-level residual prediction.

        Args:
            E_grid (torch.Tensor): The foundational grid embeddings of shape (m_cells, d).
            grid_edge_index (torch.Tensor): Edge indices defining adjacency between grid cells.
            H_region (torch.Tensor): Enriched region embeddings of shape (n_regions, d).
            target_indices (torch.Tensor): Indices identifying the specific grids in the current batch.
            grid_to_batch_idx (torch.Tensor): Mapping array linking each target grid to its parent region.

        Returns:
            torch.Tensor: The predicted residual values for the target grids, of shape (num_target_grids,).
        """
        E_grid_norm = self.input_norm(E_grid)

        X_grid_local = self.local_agg(E_grid_norm, grid_edge_index) 
        X_grid_batch = X_grid_local[target_indices]
        
        X_grid_batch = self.grid_norm(X_grid_batch)
        
        parent_region_feat = H_region[grid_to_batch_idx]
        parent_region_feat = self.region_norm(parent_region_feat)
        
        Z = torch.cat([X_grid_batch, parent_region_feat], dim=-1)
        
        grid_residual = self.mlp(Z)
        
        return grid_residual.squeeze(-1)