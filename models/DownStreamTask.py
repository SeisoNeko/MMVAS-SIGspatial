import torch.nn as nn
from models.RegionEnricher import RegionEnricher
from models.LocalGridPredictor import LocalGridPredictor
from torch_geometric.utils import dropout_edge

class DownstreamTaskModel(nn.Module):
    """
    Wraps Phase 3 & 4: Region Enrichment and Local Grid Prediction.
    This allows us to train the final downstream task end-to-end.
    """
    def __init__(self, embed_dim, dropout_rate=0.2):
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
        # 1. Macro Pathway (The Village Baseline)
        H_enriched = self.enricher(batch_H, sv_images, raw_tax_sequences)
        region_base_value = self.macro_head(H_enriched).view(-1)

        dropped_edge_index, _ = dropout_edge(grid_edge_index, p=self.edge_dropout_rate, force_undirected=True, training=self.training)
        # 2. Micro Pathway (The Grid Residual)
        grid_residual = self.predictor(E_grid, dropped_edge_index, H_enriched, target_indices, grid_to_batch_idx).view(-1)

        expanded_baseline = region_base_value[grid_to_batch_idx]
        final_preds = expanded_baseline.detach() + grid_residual
        return final_preds, region_base_value