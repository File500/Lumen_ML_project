import torch
import torch.nn as nn
from torchvision import models

class SkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SkinToneClassifier, self).__init__()
        # Use a pre-trained ResNet as the backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)