import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class LocalGridPredictorNoRegion(nn.Module):
    """
    Ablated Downstream Model: Predicts land value using ONLY micro-level grid features.
    No macro-regional baseline, no street view, no tax history.
    """
    def __init__(self, d=144):
        super().__init__()
        # Message passing to let grids communicate with their immediate neighbors
        # 1. The FP32 Shield: Pre-normalizes raw E_grid before GATConv
        self.input_norm = nn.LayerNorm(d)
        
        # 2. Local Aggregation
        self.local_agg = GATConv(d, d, add_self_loops=True)
        
        # 3. Dual LayerNorms for stable feature concatenation
        self.grid_norm = nn.LayerNorm(d)
        
        # 4. Residual MLP Predictor
        self.mlp = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, E_grid, grid_edge_index, target_indices):
        E_grid_norm = self.input_norm(E_grid)
        # 1. Local Aggregation
        # Shape: (m_cells, d)
        X_grid_local = self.local_agg(E_grid_norm, grid_edge_index) 
        X_grid_batch = X_grid_local[target_indices]
        
        # 2. Normalize local features
        X_grid_batch = self.grid_norm(X_grid_batch)
        
        # 5. Final Prediction (Grid Residual)
        grid_residual = self.mlp(X_grid_batch)
        
        return grid_residual.squeeze(-1) # Shape: (num_target_grids,)