import torch.nn as nn
import torch

class FusionSimilarityLoss(nn.Module):
    """
    Maximizes the similarity between individual intra-view feature embeddings 
    and the final DAFusion embedding, while strictly preventing dimensional collapse.
    """
    def __init__(self, penalty_weight=2.0):
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss()
        self.penalty_weight = penalty_weight

    def forward(self, intra_view_list, E):
        """
        Args:
            intra_view_list: A list of tensors [z_poi, z_lu, z_gn, z_sat], 
                             each of shape (batch_size, embed_dim)
            E: The final fused embedding matrix of shape (batch_size, embed_dim)
        """
        batch_size = E.shape[0]
        target = torch.ones(batch_size, device=E.device)
        
        total_similarity_loss = 0.0
        
        # 1. Pull views together
        for z_view in intra_view_list:
            loss = self.criterion(z_view, E, target)
            total_similarity_loss += loss
        avg_similarity_loss = total_similarity_loss / len(intra_view_list)
            
        # 2. Push different grids apart (Anti-Collapse)
        E_norm = torch.nn.functional.normalize(E, p=2, dim=1)
        sim_matrix = torch.matmul(E_norm, E_norm.T)
        
        off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=E.device)
        
        # Squaring the matrix ensures both highly positive AND highly negative 
        # correlations are penalized, forcing grids to have unique/orthogonal features.
        collapse_penalty = (sim_matrix[off_diag_mask] ** 2).mean()
        
        return avg_similarity_loss + (self.penalty_weight * collapse_penalty)
    
class CustomMSELoss(nn.Module):
    """
    Custom MSE Loss that heavily penalizes errors on extreme high AND extreme low targets.
    Designed to break 'Regression to the Mean' in imbalanced regression tasks.
    """
    def __init__(self, z_threshold=1.0, tail_weight=5.0):
        super().__init__()
        self.z_threshold = z_threshold 
        self.tail_weight = tail_weight

    def forward(self, predictions, targets):
        sq_error = (predictions - targets) ** 2
        
        tail_mask = (torch.abs(targets) >= self.z_threshold).float()
        
        weights = 1.0 + (tail_mask * (self.tail_weight - 1.0))

        weighted_loss = (sq_error * weights).mean()
        return weighted_loss