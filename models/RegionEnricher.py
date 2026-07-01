import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

class TRAlign(nn.Module):  # Abandon
    """
    Text-Region Alignment Module:
    Extracts task-relevant geographic knowledge from LLM embeddings using dimension-wise similarity.
    Update: Abandon due to moving text processing to GridLearner.
    """
    def __init__(self, d=144, d_llm=4096):
        super().__init__()
        # Linear projections for the Query, Key, and Value matrices
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d_llm, d_llm, bias=False)
        self.W_V = nn.Linear(d_llm, d_llm, bias=False) 
        
        # Final MLP to stabilize dimensionality after residual connection
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        
    def forward(self, H, H_t):
        """
        Args:
            H: Geometric region embeddings, shape (n_regions, d)
            H_t: LLM textual region embeddings, shape (n_regions, d_llm)
        """
        # Form Queries, Keys, and Values
        Q_R = self.W_Q(H)      # Shape: (n_regions, d)
        K_T = self.W_K(H_t)    # Shape: (n_regions, d_llm)
        V_T = self.W_V(H_t)    # Shape: (n_regions, d_llm)
        
        # Dimension-wise Similarity Matrix M^t
        # Formula: M^t = Softmax((W_Q H)^T (W_K H^t))
        # Q_R.T is (d, n_regions) and K_T is (n_regions, d_llm)
        M_t = torch.matmul(Q_R.t(), K_T) # Shape: (d, d_llm)
        M_t = F.softmax(M_t, dim=-1)
        
        # Retrieve textual information and transpose M_t for alignment
        # Formula: (W_V H^t) (M^t)^T
        # V_T is (n_regions, d_llm) and M_t.T is (d_llm, d)
        retrieved_text = torch.matmul(V_T, M_t.t()) # Shape: (n_regions, d)
        
        # Residual addition and MLP projection
        H_t_hat = self.mlp(retrieved_text + H)
        
        return H_t_hat
    
class SVRAlign(nn.Module):
    """
    Street View-Region Alignment Module:
    Extracts task-relevant ground-level visual features through cross-attention.
    """
    def __init__(self, d=144, d_img=768, d_proj=256):
        super().__init__()
        # Project region embedding to shared latent space
        self.W_Q = nn.Linear(d, d_proj, bias=False)
        # Project image embeddings to shared latent space
        self.W_K = nn.Linear(d_img, d_proj, bias=False)
        # Project image values to the region embedding dimension d
        self.W_V = nn.Linear(d_img, d, bias=False) 
        
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.d_proj = d_proj
        
    def forward(self, H, U):
        """
        Args:
            H: Geometric region embeddings, shape (n_regions, d)
            U: Street view image embeddings, shape (n_regions, 64, d_img)
        """
        # H acts as the querying vector (n_regions, 1, d_proj)
        Q = self.W_Q(H).unsqueeze(1) 
        
        # U acts as keys and values
        K = self.W_K(U) # Shape: (n_regions, 64, d_proj)
        V = self.W_V(U) # Shape: (n_regions, 64, d)
        
        # Scaled dot-product attention
        # Q * K^T -> (n_regions, 1, d_proj) x (n_regions, d_proj, 64) -> (n_regions, 1, 64)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_proj ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of visual values
        # (n_regions, 1, 64) x (n_regions, 64, d) -> (n_regions, 1, d)
        context = torch.bmm(attn_weights, V).squeeze(1) # Shape: (n_regions, d)
        
        H_sv_hat = self.mlp(context)
        return H_sv_hat
    
class TaxRAlign(nn.Module):
    """
    Tax Registration-Region Alignment Module:
    Extracts task-relevant temporal economic features through cross-attention.
    """
    def __init__(self, d=144, d_tax=128, d_proj=256):
        super().__init__()
        # Project static region embedding to shared latent space (Query)
        self.W_Q = nn.Linear(d, d_proj, bias=False)
        # Project temporal tax embeddings to shared latent space (Keys)
        self.W_K = nn.Linear(d_tax, d_proj, bias=False)
        # Project temporal tax values to the region embedding dimension d (Values)
        self.W_V = nn.Linear(d_tax, d, bias=False) 
        
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.d_proj = d_proj
        
    def forward(self, H, T):
        """
        Args:
            H: Geometric region embeddings, shape (n_regions, d)
            T: Temporal tax embeddings (e.g., from an LSTM), shape (n_regions, time_steps, d_tax)
        """
        # H acts as the querying vector. Shape becomes (n_regions, 1, d_proj)
        Q = self.W_Q(H).unsqueeze(1) 
        
        # T acts as keys and values
        K = self.W_K(T) # Shape: (n_regions, time_steps, d_proj)
        V = self.W_V(T) # Shape: (n_regions, time_steps, d)
        
        # Scaled dot-product cross-attention
        # Q * K^T -> (n_regions, 1, d_proj) x (n_regions, d_proj, time_steps) -> (n_regions, 1, time_steps)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_proj ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of temporal values based on attention scores
        # (n_regions, 1, time_steps) x (n_regions, time_steps, d) -> (n_regions, 1, d)
        context = torch.bmm(attn_weights, V).squeeze(1) # Shape: (n_regions, d)
        
        # Final projection to stabilize dimensions
        H_tax_hat = self.mlp(context)
        
        return H_tax_hat
    
class RegionEnricher(nn.Module):
    """
    Stage 3 Integration Module:
    Enriches the aggregated geometric region representations with Visual (SV) and Temporal (Tax) data.
    """
    def __init__(self, d=144, d_img=2048, raw_tax_dim=3, d_tax=128, d_proj=256):
        super().__init__()

        # Initialize and Freeze the ResNet50 Backbone
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity() # type: ignore # Remove the final classification layer
        self.resnet.eval() # Set to evaluation mode
        for param in self.resnet.parameters():
            param.requires_grad = False # Freeze parameters

        # Temporal Encoder for Tax Data
        self.tax_lstm = nn.LSTM(input_size=raw_tax_dim, hidden_size=d_tax, 
                                num_layers=1, batch_first=True)

        # Initialize Alignment Modules
        # self.t_ralign = TRAlign(d=d, d_llm=d_llm)
        self.sv_ralign = SVRAlign(d=d, d_img=d_img, d_proj=d_proj)
        self.tax_ralign = TaxRAlign(d=d, d_tax=d_tax, d_proj=d_proj)
        
        # The FFN takes the concatenation of H, H^sv_hat, and H^tax_hat (3 * d)
        self.enrichment_fusion = nn.Sequential(
            nn.Linear(3 * d, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2), # Dropout added for regularization during task training
            nn.Linear(256, d)
        )

    def extract_visual_embeddings(self, sv_images):
        """Extracts visual vectors for the # images per region."""
        # sv_images shape: (n_regions, #, 3, H, W)
        n_regions, num_images, c, h, w = sv_images.shape
        
        # Flatten the batch to process all images through ResNet simultaneously
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
        """
        Args:
            H: Geometric region embeddings from AdaRegionGen (n_regions, d)
            input_ids: Tokenized geographic text descriptions (n_regions, seq_len)
            attention_mask: Attention mask for the tokenized text (n_regions, seq_len)
            sv_images: Batch of sampled street view images (n_regions, 10, 3, H, W)
            raw_tax_sequences: Time-series tax data (n_regions, time_steps, raw_tax_dim)
        """
        # Pass raw data through frozen backbones
        U = self.extract_visual_embeddings(sv_images)
        T, (h_n, c_n) = self.tax_lstm(raw_tax_sequences) # T shape: (n_regions, time_steps, d_tax)
        
        # Semantic and Visual Alignment Enhancement
        H_sv_enhanced = self.sv_ralign(H, U)
        H_tax_enhanced = self.tax_ralign(H, T)
        # Concatenation and Prediction
        final_representation = torch.cat([H, H_sv_enhanced, H_tax_enhanced], dim=1)
        H_enriched = self.enrichment_fusion(final_representation)
        
        return H_enriched # Shape: (n_regions, d)
