import torch.nn as nn
import torch

class FusionSimilarityLoss(nn.Module):
    """Calculates the similarity loss between intra-view features and the fused embedding.

    This loss function maximizes the cosine similarity between individual view 
    embeddings and the final fused embedding, while incorporating a penalty term 
    to prevent dimensional collapse by encouraging orthogonality across different 
    samples in the batch.
    """

    def __init__(self, penalty_weight=2.0):
        """Initializes the FusionSimilarityLoss module.

        Args:
            penalty_weight (float, optional): The scaling factor for the dimensional 
                collapse penalty term. Defaults to 2.0.
        """
        super().__init__()
        self.criterion = nn.CosineEmbeddingLoss()
        self.penalty_weight = penalty_weight

    def forward(self, intra_view_list, E):
        """Computes the combined similarity and anti-collapse loss.

        Args:
            intra_view_list (list of torch.Tensor): A list containing individual view 
                embedding tensors, each of shape (batch_size, embed_dim).
            E (torch.Tensor): The final fused embedding matrix of shape 
                (batch_size, embed_dim).

        Returns:
            torch.Tensor: The computed scalar loss value.
        """
        batch_size = E.shape[0]
        target = torch.ones(batch_size, device=E.device)
        
        total_similarity_loss = 0.0
        
        for z_view in intra_view_list:
            loss = self.criterion(z_view, E, target)
            total_similarity_loss += loss
        avg_similarity_loss = total_similarity_loss / len(intra_view_list)
            
        E_norm = torch.nn.functional.normalize(E, p=2, dim=1)
        sim_matrix = torch.matmul(E_norm, E_norm.T)
        
        off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=E.device)
        
        collapse_penalty = (sim_matrix[off_diag_mask] ** 2).mean()
        
        return avg_similarity_loss + (self.penalty_weight * collapse_penalty)
    
class CustomMSELoss(nn.Module):
    """Applies a weighted Mean Squared Error loss focusing on distribution tails.

    This custom loss function disproportionately penalizes prediction errors on 
    extreme target values (both high and low) to mitigate the regression-to-the-mean 
    effect often observed in imbalanced regression datasets.
    """

    def __init__(self, z_threshold=1.0, tail_weight=5.0):
        """Initializes the CustomMSELoss module.

        Args:
            z_threshold (float, optional): The standardized threshold beyond which 
                a target is considered part of the tail distribution. Defaults to 1.0.
            tail_weight (float, optional): The multiplier applied to the squared 
                error for targets within the tail distribution. Defaults to 5.0.
        """
        super().__init__()
        self.z_threshold = z_threshold 
        self.tail_weight = tail_weight

    def forward(self, predictions, targets):
        """Computes the weighted Mean Squared Error loss.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth target values.

        Returns:
            torch.Tensor: The computed scalar weighted loss.
        """
        sq_error = (predictions - targets) ** 2
        
        tail_mask = (torch.abs(targets) >= self.z_threshold).float()
        
        weights = 1.0 + (tail_mask * (self.tail_weight - 1.0))

        weighted_loss = (sq_error * weights).mean()
        return weighted_loss