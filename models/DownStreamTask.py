import torch.nn as nn
from models.RegionEnricher import RegionEnricher
from models.LocalGridPredictor import LocalGridPredictor
from torch_geometric.utils import dropout_edge

class DownstreamTaskModel(nn.Module):
    """Wraps the Region Enrichment and Local Grid Prediction phases.

    This module acts as the final downstream task model, enabling end-to-end 
    training by combining macroscopic regional features with microscopic grid residuals.
    """

    def __init__(self, embed_dim, dropout_rate=0.2):
        """Initializes the DownstreamTaskModel.

        Args:
            embed_dim (int): The dimensionality of the input node embeddings.
            dropout_rate (float, optional): The dropout probability applied to edges 
                during the forward pass. Defaults to 0.2.
        """
        super().__init__()
        self.enricher = RegionEnricher(d=embed_dim, d_img=2048, raw_tax_dim=3, d_tax=128, d_proj=256)
        self.predictor = LocalGridPredictor(d=embed_dim)
        self.macro_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.edge_dropout_rate = dropout_rate

    def forward(self, E_grid, grid_edge_index, batch_H, sv_images, raw_tax_sequences, target_indices, grid_to_batch_idx):
        """Executes the forward pass of the downstream prediction task.

        Args:
            E_grid (torch.Tensor): The grid cell embeddings.
            grid_edge_index (torch.Tensor): The edge indices defining the grid connectivity.
            batch_H (torch.Tensor): The batched macroscopic region embeddings.
            sv_images (torch.Tensor): The street-view image tensors for the regions.
            raw_tax_sequences (torch.Tensor): The historical tax data sequences.
            target_indices (torch.Tensor): Indices mapping to the specific target grids.
            grid_to_batch_idx (torch.Tensor): Tensor mapping individual grids to their 
                corresponding region in the current batch.

        Returns:
            tuple: A tuple containing:
                - final_preds (torch.Tensor): The final computed predictions (baseline + residual).
                - region_base_value (torch.Tensor): The macroscopic baseline predictions per region.
        """
        H_enriched = self.enricher(batch_H, sv_images, raw_tax_sequences)
        region_base_value = self.macro_head(H_enriched).view(-1)

        dropped_edge_index, _ = dropout_edge(
            grid_edge_index, 
            p=self.edge_dropout_rate, 
            force_undirected=True, 
            training=self.training
        )
        
        grid_residual = self.predictor(
            E_grid, 
            dropped_edge_index, 
            H_enriched, 
            target_indices, 
            grid_to_batch_idx
        ).view(-1)

        expanded_baseline = region_base_value[grid_to_batch_idx]
        final_preds = expanded_baseline.detach() + grid_residual
        
        return final_preds, region_base_value