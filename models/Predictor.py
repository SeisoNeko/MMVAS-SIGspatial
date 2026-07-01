import torch
import torch.nn as nn
from transformers import AutoModel

from models.GridLearner import GridLearner
from models.AdaRegionGen import AdaRegionGen
from models.RegionEnricher import RegionEnricher
from models.LocalGridPredictor import LocalGridPredictor

class LandValuePredictor(nn.Module):
    """Orchestrates the multi-stage land value prediction architecture.

    This master module coordinates the data flow between language models, spatial 
    grid learning, dynamic region aggregation, and local grid residual prediction.
    """

    def __init__(self, input_dims, d=144, d_llm=768, d_img=2048, raw_tax_dim=3):
        """Initializes the LandValuePredictor.

        Args:
            input_dims (dict): Dictionary defining input dimensions for the spatial graphs.
            d (int, optional): The base projection dimensionality for embeddings. Defaults to 144.
            d_llm (int, optional): The dimensionality of the language model outputs. Defaults to 768.
            d_img (int, optional): The dimensionality of the visual features. Defaults to 2048.
            raw_tax_dim (int, optional): The dimensionality of the input tax sequences. Defaults to 3.
        """
        super().__init__()
        
        self.llm = AutoModel.from_pretrained("bert-base-chinese")
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False
            
        self.ada_region_gen = AdaRegionGen()

        self.grid_learner = GridLearner(input_dims=input_dims, text_dim=d_llm, embed_dim=d)
        
        self.region_enricher = RegionEnricher(d=d, d_img=d_img, raw_tax_dim=raw_tax_dim)
        
        self.local_predictor = LocalGridPredictor(d=d)

    def extract_textual_embeddings(self, input_ids, attention_mask):
        """Extracts contextualized sequence embeddings from the frozen language model.

        Args:
            input_ids (torch.Tensor): Tokenized input sequences of shape (batch_size, seq_len).
            attention_mask (torch.Tensor): Binary attention masks of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The extracted CLS token embeddings of shape (batch_size, d_llm).
        """
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            cls_embeddings = hidden_states[:, 0, :]
        return cls_embeddings

    def forward(self, 
                input_ids, attention_mask, sv_images, raw_tax_sequences,
                poi_feat, poi_edge_index, poi_edge_weights,
                lu_feat, lu_edge_index, lu_edge_weights,
                gn_feat, gn_edge_index, farbac_feat, farbac_edge_index, farbac_edge_weights,
                sat_features, grid_to_region_mapping, grid_local_edges, regions_gdf, cells_gdf):
        """Executes the full forward pass of the land value prediction pipeline.

        Args:
            input_ids (torch.Tensor): Tokenized region-level textual descriptions.
            attention_mask (torch.Tensor): Attention masks for the textual descriptions.
            sv_images (torch.Tensor): Street-view imagery tensors.
            raw_tax_sequences (torch.Tensor): Historical tax value sequences.
            poi_feat (torch.Tensor): Node features for the Point of Interest graph.
            poi_edge_index (torch.Tensor): Edge indices for the POI graph.
            poi_edge_weights (torch.Tensor): Edge weights for the POI graph.
            lu_feat (torch.Tensor): Node features for the Land Use graph.
            lu_edge_index (torch.Tensor): Edge indices for the Land Use graph.
            lu_edge_weights (torch.Tensor): Edge weights for the Land Use graph.
            gn_feat (torch.Tensor): Node features for the Geographic Neighbor graph.
            gn_edge_index (torch.Tensor): Edge indices for the Geographic Neighbor graph.
            farbac_feat (torch.Tensor): Node features for the FAR/BAC graph.
            farbac_edge_index (torch.Tensor): Edge indices for the FAR/BAC graph.
            farbac_edge_weights (torch.Tensor): Edge weights for the FAR/BAC graph.
            sat_features (torch.Tensor): Extracted satellite imagery feature matrices.
            grid_to_region_mapping (torch.Tensor): Index mapping from grids to regions.
            grid_local_edges (torch.Tensor): Adjacency matrix for grid-level message passing.
            regions_gdf (gpd.GeoDataFrame): Geometric data for the macro-regions.
            cells_gdf (gpd.GeoDataFrame): Geometric data for the micro-grids.

        Returns:
            torch.Tensor: The final predicted continuous land values for the grid cells.
        """
        H_t = self.extract_textual_embeddings(input_ids, attention_mask)
        
        text_emb_mapped = H_t[grid_to_region_mapping] 
        
        E_grid, _ = self.grid_learner(
            poi_feat, poi_edge_index, poi_edge_weights,
            lu_feat, lu_edge_index, lu_edge_weights,
            gn_feat, gn_edge_index, farbac_feat, farbac_edge_index, farbac_edge_weights,
            sat_features, text_emb_mapped
        )
        
        H_raw = self.ada_region_gen.forward(regions_gdf, cells_gdf, E_grid)
        
        H_enriched = self.region_enricher(H_raw, sv_images, raw_tax_sequences)
        
        land_value_preds = self.local_predictor(E_grid, grid_local_edges, H_enriched)
        
        return land_value_preds