import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.
    
    Args:
        gamma (float): Focusing parameter (default: 2.0)
        alpha (float): Balancing parameter (default: 0.25)
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): True labels
        
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Compute binary cross-entropy loss without reduction
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute probabilities to prevent numerical instability
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        
        # Compute focal loss
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        return focal_loss.mean()