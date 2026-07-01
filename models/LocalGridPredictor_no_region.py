import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LocalGridPredictorNoRegion(nn.Module):
    """Ablated downstream model predicting land value using only micro-level grid features.

    This baseline model excludes macroscopic regional information, street-view 
    imagery, and historical tax data, relying solely on local grid embeddings 
    and immediate neighbor message passing.
    """

    def __init__(self, d=144):
        """Initializes the LocalGridPredictorNoRegion module.

        Args:
            d (int, optional): The dimensionality of the input node embeddings. Defaults to 144.
        """
        super().__init__()
        
        self.input_norm = nn.LayerNorm(d)
        
        self.local_agg = GATConv(d, d, add_self_loops=True)
        
        self.grid_norm = nn.LayerNorm(d)
        
        self.mlp = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, E_grid, grid_edge_index, target_indices):
        """Executes the forward pass for the ablated grid prediction.

        Args:
            E_grid (torch.Tensor): The foundational grid embeddings of shape (m_cells, d).
            grid_edge_index (torch.Tensor): Edge indices defining adjacency between grid cells.
            target_indices (torch.Tensor): Indices identifying the specific grids in the current batch.

        Returns:
            torch.Tensor: The predicted land values for the target grids, of shape (num_target_grids,).
        """
        E_grid_norm = self.input_norm(E_grid)

        X_grid_local = self.local_agg(E_grid_norm, grid_edge_index) 
        X_grid_batch = X_grid_local[target_indices]
        
        X_grid_batch = self.grid_norm(X_grid_batch)
        
        grid_residual = self.mlp(X_grid_batch)
        
        return grid_residual.squeeze(-1)