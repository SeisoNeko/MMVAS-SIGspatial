import torch
import torch.nn as nn
from transformers import AutoModel

from models.GridLearner import GridLearner
from models.AdaRegionGen import AdaRegionGen
from models.RegionEnricher import RegionEnricher
from models.LocalGridPredictor import LocalGridPredictor

class LandValuePredictor(nn.Module):
    """
    The Master Orchestrator:
    Handles raw inputs, runs the heavy backbones (LLM), and coordinates the data flow 
    between the Grid, Region, and Prediction stages.
    """
    def __init__(self, input_dims, d=144, d_llm=768, d_img=2048, raw_tax_dim=3):
        super().__init__()
        
        # -----------------------------------------
        # Stage 0: The Heavy Backbones (Text)
        # -----------------------------------------
        self.llm = AutoModel.from_pretrained("bert-base-chinese")
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False
            
        self.ada_region_gen = AdaRegionGen()

        # -----------------------------------------
        # Stages 1 & 2: Early Semantic Grid Learning
        # -----------------------------------------
        self.grid_learner = GridLearner(input_dims=input_dims, text_dim=d_llm, embed_dim=d)
        
        # -----------------------------------------
        # Stage 3: Regional Enrichment
        # -----------------------------------------
        self.region_enricher = RegionEnricher(d=d, d_img=d_img, raw_tax_dim=raw_tax_dim)
        
        # -----------------------------------------
        # Stages 4 & 5: Cross-Attention & Prediction
        # -----------------------------------------
        self.local_predictor = LocalGridPredictor(d=d)

    def extract_textual_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            cls_embeddings = hidden_states[:, 0, :]
        return cls_embeddings # Shape: (n_regions, d_llm)

    def forward(self, 
                input_ids, attention_mask, sv_images, raw_tax_sequences,
                poi_feat, poi_edge_index, poi_edge_weights,
                lu_feat, lu_edge_index, lu_edge_weights,
                gn_feat, gn_edge_index, farbac_feat, farbac_edge_index, farbac_edge_weights,
                sat_features, grid_to_region_mapping, grid_local_edges, regions_gdf, cells_gdf):
        
        # 1. Process Text (Region-level)
        # H_t shape: (n_regions, d_llm)
        H_t = self.extract_textual_embeddings(input_ids, attention_mask)
        
        # 2. Map Region Text Embeddings to Grids for FiLM
        # grid_to_region_mapping is a tensor of shape (m_cells,) containing the region index for each grid cell.
        # This duplicates the region's text embedding for every grid cell inside that region.
        text_emb_mapped = H_t[grid_to_region_mapping] # Shape: (m_cells, d_llm)
        
        # 3. Stage 1 & 2: GridLearner with Early Semantic Injection
        # E_grid shape: (m_cells, d)
        E_grid, _ = self.grid_learner(
            poi_feat, poi_edge_index, poi_edge_weights,
            lu_feat, lu_edge_index, lu_edge_weights,
            gn_feat, gn_edge_index, farbac_feat, farbac_edge_index, farbac_edge_weights,
            sat_features, text_emb_mapped
        )
        
        # 4. Aggregate Grids into Raw Regions (AdaRegionGen)
        # H_raw shape: (n_regions, d)
        H_raw = self.ada_region_gen.forward(regions_gdf, cells_gdf, E_grid)
        
        # 5. Stage 3: Region Enrichment
        # H_enriched shape: (n_regions, d)
        H_enriched = self.region_enricher(H_raw, sv_images, raw_tax_sequences)
        
        # 6. Stage 4 & 5: Local Grid Prediction
        # Final output shape: (m_cells,)
        land_value_preds = self.local_predictor(E_grid, grid_local_edges, H_enriched)
        
        return land_value_preds