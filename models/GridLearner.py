import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision.models import resnet50, ResNet50_Weights

class FiLMLayer(nn.Module):
    """Early Semantic Conditioning: Modulates spatial features based on text prompts."""
    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.gamma = nn.Linear(text_dim, feature_dim)
        self.beta = nn.Linear(text_dim, feature_dim)

    def forward(self, x, text_emb):
        # Apply text-based affine transformation
        return self.gamma(text_emb) * x + self.beta(text_emb)

class GATBranch(nn.Module):
    """Processes structured spatial graphs (POI, Land Use, Geographic Neighbors)"""
    def __init__(self, in_channels, out_channels, text_dim=768, num_layers=2):
        super().__init__()
        self.film = FiLMLayer(text_dim=text_dim, feature_dim=in_channels)
        self.layers = nn.ModuleList()
        # Initial input to latent projection
        self.layers.append(GATConv(in_channels, out_channels, add_self_loops=True))
        
        # Hidden GAT layers
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_channels, out_channels, add_self_loops=True))

    def forward(self, x, edge_index, text_emb, edge_attr=None):
        x = self.film(x, text_emb)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x) # LeakyReLU as defined in the architecture
        return x

class CNNBranch(nn.Module):
    """Processes unstructured visual data (Satellite Imagery)"""
    def __init__(self, out_channels=144):
        super().__init__()
        # Load pre-trained ResNet50 backbone
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the final classification layer to extract raw visual feature maps
        self.resnet.fc = nn.Identity()
        
        # MLP bottleneck to align visual dimensionality with the graph branches
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )

    def forward(self, x):
        # x shape: (batch_size, 3, H, W)
        x = self.resnet(x)
        x = self.mlp(x)
        return x

class InterViewAttention(nn.Module):
    def __init__(self, embed_dim=144):
        super().__init__()
        # 1-layer Transformer Encoder acting as multi-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Learnable residual scalar to stabilize training
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, Z):
        # Input Z shape: (m_cells, num_views, embed_dim)
        attn_out = self.self_attention(Z)
        
        # Residual connection weighted by beta
        Z_updated = Z + self.beta * attn_out
        return Z_updated
    
class DAFusion(nn.Module):
    def __init__(self, embed_dim=144, num_views=5):
        super().__init__()
        # ViewFusion adaptive scalar weights
        self.gamma = nn.Parameter(torch.ones(num_views) / num_views)

        # RegionFusion Spatial Transformer Encoder (STE)
        ste_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.ste = nn.TransformerEncoder(ste_layer, num_layers=3) # Typically 3 layers
        
        # Final lightweight projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, Z):
        # Z shape: (m_cells, num_views, embed_dim)
        
        # --- ViewFusion ---
        # Normalize weights so they sum to 1
        gamma_norm = F.softmax(self.gamma, dim=0) 
        # Weighted summation of views
        Z_fused = torch.sum(Z * gamma_norm.view(1, -1, 1), dim=1) # Shape: (m_cells, embed_dim)
        
        # --- RegionFusion ---
        # Introduce a dummy batch dimension so the Transformer processes the m_cells as a "sequence"
        Z_fused = Z_fused.unsqueeze(0) # Shape: (1, m_cells, embed_dim)
        E_prime = self.ste(Z_fused)
        
        # Remove dummy batch dim and pass through final MLP
        E_prime = E_prime.squeeze(0)
        E = self.mlp(E_prime) # Shape: (m_cells, embed_dim)
        
        return E
    
class GridLearner(nn.Module):
    def __init__(self, input_dims, text_dim=768, embed_dim=144):
        super().__init__()
        # 1. Initialize Intra-View feature branches
        self.gat_poi = GATBranch(in_channels=input_dims['poi'], out_channels=embed_dim, text_dim=text_dim)
        self.gat_lu = GATBranch(in_channels=input_dims['land_use'], out_channels=embed_dim, text_dim=text_dim)
        self.gat_gn = GATBranch(in_channels=input_dims['neighbor'], out_channels=embed_dim, text_dim=text_dim)
        self.gat_farbac = GATBranch(in_channels=input_dims['farbac'], out_channels=embed_dim, text_dim=text_dim)
        self.sat_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
        # 2. Initialize Inter-View Attention
        self.inter_view = InterViewAttention(embed_dim=embed_dim)
        
        # 3. Initialize DAFusion
        self.da_fusion = DAFusion(embed_dim=embed_dim, num_views=5)

    def forward(self, 
                poi_feat, poi_edge_index, poi_edge_weights,
                lu_feat, lu_edge_index, lu_edge_weights,
                gn_feat, gn_edge_index, farbac_feat, farbac_edge_index, farbac_edge_weights,
                sat_features, text_embeddings_mapped):
        
        # Phase 1: Intra-view learning
        # Note: Depending on memory, these operations can be executed concurrently
        z_poi = self.gat_poi(poi_feat, poi_edge_index, text_embeddings_mapped, edge_attr=poi_edge_weights)
        z_lu = self.gat_lu(lu_feat, lu_edge_index, text_embeddings_mapped, edge_attr=lu_edge_weights)
        z_gn = self.gat_gn(gn_feat, gn_edge_index, text_embeddings_mapped)
        z_farbac = self.gat_farbac(farbac_feat, farbac_edge_index, text_embeddings_mapped, edge_attr=farbac_edge_weights)
        z_sat = self.sat_projector(sat_features)
        
        # Phase 2: Inter-view stack and attention
        # Shape becomes (m_cells, 5, embed_dim)
        Z_stacked = torch.stack([z_poi, z_lu, z_gn, z_farbac, z_sat], dim=1)
        Z_attended = self.inter_view(Z_stacked)
        
        # Phase 3: DAFusion block
        # Outputs the final general purpose embedding matrix E
        E = self.da_fusion(Z_attended) 
        
        return E, [z_poi, z_lu, z_gn, z_farbac, z_sat]
