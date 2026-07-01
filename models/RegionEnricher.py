import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

class TRAlign(nn.Module):
    """Deprecated Text-Region Alignment Module.

    Originally designed to extract task-relevant geographic knowledge from 
    language model embeddings using dimension-wise similarity. Text processing 
    has since been delegated to the GridLearner module.
    """

    def __init__(self, d=144, d_llm=4096):
        """Initializes the TRAlign module.

        Args:
            d (int, optional): Dimensionality of the region embeddings. Defaults to 144.
            d_llm (int, optional): Dimensionality of the language model embeddings. Defaults to 4096.
        """
        super().__init__()
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d_llm, d_llm, bias=False)
        self.W_V = nn.Linear(d_llm, d_llm, bias=False) 
        
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        
    def forward(self, H, H_t):
        """Executes the forward pass for text-region alignment.

        Args:
            H (torch.Tensor): Geometric region embeddings of shape (n_regions, d).
            H_t (torch.Tensor): Textual region embeddings of shape (n_regions, d_llm).

        Returns:
            torch.Tensor: The text-aligned region embeddings.
        """
        Q_R = self.W_Q(H)      
        K_T = self.W_K(H_t)    
        V_T = self.W_V(H_t)    
        
        M_t = torch.matmul(Q_R.t(), K_T) 
        M_t = F.softmax(M_t, dim=-1)
        
        retrieved_text = torch.matmul(V_T, M_t.t()) 
        
        H_t_hat = self.mlp(retrieved_text + H)
        
        return H_t_hat
    
class SVRAlign(nn.Module):
    """Street View-Region Alignment Module.

    Extracts task-relevant ground-level visual features through cross-attention 
    between macro-region embeddings and local street-view imagery.
    """

    def __init__(self, d=144, d_img=768, d_proj=256):
        """Initializes the SVRAlign module.

        Args:
            d (int, optional): Dimensionality of the region embeddings. Defaults to 144.
            d_img (int, optional): Dimensionality of the visual features. Defaults to 768.
            d_proj (int, optional): Dimensionality of the shared latent projection space. Defaults to 256.
        """
        super().__init__()
        self.W_Q = nn.Linear(d, d_proj, bias=False)
        self.W_K = nn.Linear(d_img, d_proj, bias=False)
        self.W_V = nn.Linear(d_img, d, bias=False) 
        
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.d_proj = d_proj
        
    def forward(self, H, U):
        """Executes the forward pass for street view-region alignment.

        Args:
            H (torch.Tensor): Geometric region embeddings of shape (n_regions, d).
            U (torch.Tensor): Street view image embeddings of shape (n_regions, num_images, d_img).

        Returns:
            torch.Tensor: The visually enhanced region embeddings.
        """
        Q = self.W_Q(H).unsqueeze(1) 
        
        K = self.W_K(U) 
        V = self.W_V(U) 
        
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_proj ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.bmm(attn_weights, V).squeeze(1) 
        
        H_sv_hat = self.mlp(context)
        return H_sv_hat
    
class TaxRAlign(nn.Module):
    """Tax Registration-Region Alignment Module.

    Extracts task-relevant temporal economic features through cross-attention 
    between static geometric regions and sequential historical tax data.
    """

    def __init__(self, d=144, d_tax=128, d_proj=256):
        """Initializes the TaxRAlign module.

        Args:
            d (int, optional): Dimensionality of the region embeddings. Defaults to 144.
            d_tax (int, optional): Dimensionality of the temporal tax embeddings. Defaults to 128.
            d_proj (int, optional): Dimensionality of the shared latent projection space. Defaults to 256.
        """
        super().__init__()
        self.W_Q = nn.Linear(d, d_proj, bias=False)
        self.W_K = nn.Linear(d_tax, d_proj, bias=False)
        self.W_V = nn.Linear(d_tax, d, bias=False) 
        
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.d_proj = d_proj
        
    def forward(self, H, T):
        """Executes the forward pass for tax data-region alignment.

        Args:
            H (torch.Tensor): Geometric region embeddings of shape (n_regions, d).
            T (torch.Tensor): Temporal tax embeddings of shape (n_regions, time_steps, d_tax).

        Returns:
            torch.Tensor: The economically enhanced region embeddings.
        """
        Q = self.W_Q(H).unsqueeze(1) 
        
        K = self.W_K(T) 
        V = self.W_V(T) 
        
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_proj ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.bmm(attn_weights, V).squeeze(1) 
        
        H_tax_hat = self.mlp(context)
        
        return H_tax_hat
    
class RegionEnricher(nn.Module):
    """Integrates visual and temporal modalities into macro-region representations.

    This module enriches the aggregated geometric region representations using 
    ground-level visual data (street-view imagery) and temporal economic data (tax records).
    """

    def __init__(self, d=144, d_img=2048, raw_tax_dim=3, d_tax=128, d_proj=256):
        """Initializes the RegionEnricher module.

        Args:
            d (int, optional): Dimensionality of the region embeddings. Defaults to 144.
            d_img (int, optional): Dimensionality of the extracted visual features. Defaults to 2048.
            raw_tax_dim (int, optional): Input dimensionality of the raw tax sequences. Defaults to 3.
            d_tax (int, optional): Hidden dimensionality for the LSTM tax encoder. Defaults to 128.
            d_proj (int, optional): Dimensionality of the shared latent projection spaces. Defaults to 256.
        """
        super().__init__()

        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity() 
        self.resnet.eval() 
        for param in self.resnet.parameters():
            param.requires_grad = False 

        self.tax_lstm = nn.LSTM(input_size=raw_tax_dim, hidden_size=d_tax, 
                                num_layers=1, batch_first=True)

        self.sv_ralign = SVRAlign(d=d, d_img=d_img, d_proj=d_proj)
        self.tax_ralign = TaxRAlign(d=d, d_tax=d_tax, d_proj=d_proj)
        
        self.enrichment_fusion = nn.Sequential(
            nn.Linear(3 * d, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(256, d)
        )

    def extract_visual_embeddings(self, sv_images):
        """Extracts latent feature vectors from batches of street-view images.

        Args:
            sv_images (torch.Tensor): Image tensor of shape (n_regions, num_images, c, h, w).

        Returns:
            torch.Tensor: The extracted visual features of shape (n_regions, num_images, d_img).
        """
        n_regions, num_images, c, h, w = sv_images.shape
        
        sv_images_flat = sv_images.view(-1, c, h, w)

        chunk_size = 8
        visual_features_list = []
        
        with torch.no_grad():
            for i in range(0, sv_images_flat.size(0), chunk_size):
                chunk = sv_images_flat[i:i+chunk_size]
                chunk_feat = self.resnet(chunk) 
                visual_features_list.append(chunk_feat)
            
        visual_features = torch.cat(visual_features_list, dim=0)
        U = visual_features.view(n_regions, num_images, -1)
        return U
        
    def forward(self, H, sv_images, raw_tax_sequences):
        """Executes the forward pass for regional multi-modal enrichment.

        Args:
            H (torch.Tensor): Geometric region embeddings of shape (n_regions, d).
            sv_images (torch.Tensor): Sampled street view images of shape (n_regions, num_images, 3, H, W).
            raw_tax_sequences (torch.Tensor): Time-series tax data of shape (n_regions, time_steps, raw_tax_dim).

        Returns:
            torch.Tensor: The final, fully enriched region representations of shape (n_regions, d).
        """
        U = self.extract_visual_embeddings(sv_images)
        T, _ = self.tax_lstm(raw_tax_sequences) 
        
        H_sv_enhanced = self.sv_ralign(H, U)
        H_tax_enhanced = self.tax_ralign(H, T)

        final_representation = torch.cat([H, H_sv_enhanced, H_tax_enhanced], dim=1)
        H_enriched = self.enrichment_fusion(final_representation)
        
        return H_enriched