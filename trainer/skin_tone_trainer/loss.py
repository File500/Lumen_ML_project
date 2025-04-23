import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalOrdinalWassersteinLoss(nn.Module):
    def __init__(self, num_classes=7, alpha=0.25, gamma=2.0, 
                 ordinal_weight=0.3, wasserstein_weight=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Focal loss parameter
        self.gamma = gamma  # Focal loss parameter
        self.ordinal_weight = ordinal_weight  # Weight for ordinal component
        self.wasserstein_weight = wasserstein_weight  # Weight for Wasserstein component
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Base loss with smoothing
        
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Basic cross-entropy with label smoothing
        ce_loss = self.ce_loss(inputs, targets)  # This is already a scalar
        
        # Focal component - focus training on hard examples
        probs = F.softmax(inputs, dim=1)
        pt = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (self.alpha * focal_weight * ce_loss).mean()  # Ensure scalar output
        
        # Explicit ordinal penalty - penalize predictions based on distance
        ordinal_loss = torch.tensor(0.0, device=device)  # Initialize as scalar tensor
        for i in range(batch_size):
            true_class = targets[i].item()
            for j in range(self.num_classes):
                if j != true_class:
                    # Penalize based on distance between classes
                    distance = abs(j - true_class)
                    ordinal_loss += distance * probs[i, j]
        
        ordinal_loss = ordinal_loss / batch_size  # This is a scalar
        
        # Wasserstein Loss component for ordinal relationships
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Calculate cumulative distributions
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(targets_one_hot, dim=1)
        
        # Earth Mover's Distance
        wasserstein_loss = torch.abs(pred_cdf - target_cdf).sum(dim=1).mean()  # This is a scalar
        
        # Combined loss with all three components
        total_loss = focal_loss + self.ordinal_weight * ordinal_loss + self.wasserstein_weight * wasserstein_loss
        
        return total_loss