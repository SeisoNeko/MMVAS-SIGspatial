import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchvision.models import resnet50, ResNet50_Weights

class FiLMLayer(nn.Module):
    """Modulates spatial features based on language semantics.

    This layer acts as an early semantic conditioning mechanism, applying an 
    affine transformation to visual or graph features using text prompt embeddings.
    """

    def __init__(self, text_dim, feature_dim):
        """Initializes the FiLMLayer.

        Args:
            text_dim (int): Dimensionality of the text embeddings.
            feature_dim (int): Dimensionality of the spatial features.
        """
        super().__init__()
        self.gamma = nn.Linear(text_dim, feature_dim)
        self.beta = nn.Linear(text_dim, feature_dim)

    def forward(self, x, text_emb):
        """Applies text-based affine transformation to the input features.

        Args:
            x (torch.Tensor): Input spatial features.
            text_emb (torch.Tensor): Input text embeddings.

        Returns:
            torch.Tensor: The semantically modulated features.
        """
        return self.gamma(text_emb) * x + self.beta(text_emb)

class GATBranch(nn.Module):
    """Processes structured spatial graphs.

    This branch handles inputs such as Points of Interest (POI), Land Use, 
    and Geographic Neighbor networks via Graph Attention mechanisms.
    """

    def __init__(self, in_channels, out_channels, text_dim=768, num_layers=2):
        """Initializes the GATBranch.

        Args:
            in_channels (int): Input feature dimensionality per node.
            out_channels (int): Output feature dimensionality.
            text_dim (int, optional): Dimensionality of conditioning text. Defaults to 768.
            num_layers (int, optional): Number of GATConv layers. Defaults to 2.
        """
        super().__init__()
        self.film = FiLMLayer(text_dim=text_dim, feature_dim=in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, out_channels, add_self_loops=True))
        
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_channels, out_channels, add_self_loops=True))

    def forward(self, x, edge_index, text_emb, edge_attr=None):
        """Executes the forward pass for the graph branch.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph connectivity indices.
            text_emb (torch.Tensor): Text embeddings for FiLM conditioning.
            edge_attr (torch.Tensor, optional): Edge features/weights. Defaults to None.

        Returns:
            torch.Tensor: Processed graph embeddings.
        """
        x = self.film(x, text_emb)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x)
        return x

class CNNBranch(nn.Module):
    """Processes unstructured visual data from satellite imagery."""

    def __init__(self, out_channels=144):
        """Initializes the CNNBranch.

        Args:
            out_channels (int, optional): Dimensionality of the final output projection. Defaults to 144.
        """
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        
        self.mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )

    def forward(self, x):
        """Extracts and projects visual features from imagery tensors.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Projected visual embeddings.
        """
        x = self.resnet(x)
        x = self.mlp(x)
        return x

class InterViewAttention(nn.Module):
    """Aligns multi-view representations using cross-view self-attention."""

    def __init__(self, embed_dim=144):
        """Initializes the InterViewAttention module.

        Args:
            embed_dim (int, optional): Dimensionality of the view representations. Defaults to 144.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, Z):
        """Computes self-attention across the different data modalities (views).

        Args:
            Z (torch.Tensor): Stacked view embeddings of shape (m_cells, num_views, embed_dim).

        Returns:
            torch.Tensor: The updated, attention-weighted view embeddings.
        """
        attn_out = self.self_attention(Z)
        Z_updated = Z + self.beta * attn_out
        return Z_updated
    
class DAFusion(nn.Module):
    """Fuses multi-view representations into a unified spatial grid embedding."""

    def __init__(self, embed_dim=144, num_views=5):
        """Initializes the DAFusion module.

        Args:
            embed_dim (int, optional): Dimensionality of the embeddings. Defaults to 144.
            num_views (int, optional): Number of distinct spatial views. Defaults to 5.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_views) / num_views)

        ste_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.ste = nn.TransformerEncoder(ste_layer, num_layers=3)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, Z):
        """Aggregates multiple views and enforces global spatial coherence.

        Args:
            Z (torch.Tensor): The attended multi-view tensors of shape (m_cells, num_views, embed_dim).

        Returns:
            torch.Tensor: Final general-purpose grid embeddings matrix 'E'.
        """
        gamma_norm = F.softmax(self.gamma, dim=0) 
        Z_fused = torch.sum(Z * gamma_norm.view(1, -1, 1), dim=1) 
        
        Z_fused = Z_fused.unsqueeze(0) 
        E_prime = self.ste(Z_fused)
        
        E_prime = E_prime.squeeze(0)
        E = self.mlp(E_prime) 
        
        return E
    
class GridLearner(nn.Module):
    """Constructs comprehensive representations for microscopic spatial grids."""

    def __init__(self, input_dims, text_dim=768, embed_dim=144):
        """Initializes the GridLearner architecture.

        Args:
            input_dims (dict): Dictionary defining input dimensions for each spatial graph view.
            text_dim (int, optional): Dimensionality of semantic text embeddings. Defaults to 768.
            embed_dim (int, optional): Target dimensionality for projection. Defaults to 144.
        """
        super().__init__()
        self.gat_poi = GATBranch(in_channels=input_dims['poi'], out_channels=embed_dim, text_dim=text_dim)
        self.gat_lu = GATBranch(in_channels=input_dims['land_use'], out_channels=embed_dim, text_dim=text_dim)
        self.gat_gn = GATBranch(in_channels=input_dims['neighbor'], out_channels=embed_dim, text_dim=text_dim)
        self.gat_farbac = GATBranch(in_channels=input_dims['farbac'], out_channels=embed_dim, text_dim=text_dim)
        self.sat_projector = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )
        
        self.inter_view = InterViewAttention(embed_dim=embed_dim)
        self.da_fusion = DAFusion(embed_dim=embed_dim, num_views=5)

    def forward(self, 
                poi_feat, poi_edge_index, poi_edge_weights,
                lu_feat, lu_edge_index, lu_edge_weights,
                gn_feat, gn_edge_index, farbac_feat, farbac_edge_index, farbac_edge_weights,
                sat_features, text_embeddings_mapped):
        """Executes the complete grid embedding generation pipeline.

        Args:
            poi_feat (torch.Tensor): POI node features.
            poi_edge_index (torch.Tensor): POI edge indices.
            poi_edge_weights (torch.Tensor): POI edge weights.
            lu_feat (torch.Tensor): Land Use node features.
            lu_edge_index (torch.Tensor): Land Use edge indices.
            lu_edge_weights (torch.Tensor): Land Use edge weights.
            gn_feat (torch.Tensor): Geographic Neighbor node features.
            gn_edge_index (torch.Tensor): Geographic Neighbor edge indices.
            farbac_feat (torch.Tensor): FAR/BAC node features.
            farbac_edge_index (torch.Tensor): FAR/BAC edge indices.
            farbac_edge_weights (torch.Tensor): FAR/BAC edge weights.
            sat_features (torch.Tensor): Pre-extracted satellite imagery features.
            text_embeddings_mapped (torch.Tensor): Semantic text embeddings aligned to grids.

        Returns:
            tuple: A tuple containing:
                - E (torch.Tensor): Final fused embeddings of shape (m_cells, embed_dim).
                - z_list (list): List containing individual branch outputs prior to stacking.
        """
        z_poi = self.gat_poi(poi_feat, poi_edge_index, text_embeddings_mapped, edge_attr=poi_edge_weights)
        z_lu = self.gat_lu(lu_feat, lu_edge_index, text_embeddings_mapped, edge_attr=lu_edge_weights)
        z_gn = self.gat_gn(gn_feat, gn_edge_index, text_embeddings_mapped)
        z_farbac = self.gat_farbac(farbac_feat, farbac_edge_index, text_embeddings_mapped, edge_attr=farbac_edge_weights)
        z_sat = self.sat_projector(sat_features)
        
        Z_stacked = torch.stack([z_poi, z_lu, z_gn, z_farbac, z_sat], dim=1)
        Z_attended = self.inter_view(Z_stacked)
        
        E = self.da_fusion(Z_attended) 
        
        return E, [z_poi, z_lu, z_gn, z_farbac, z_sat]