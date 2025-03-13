import torch
import torch.nn as nn
from torchvision import models

class SkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SkinToneClassifier, self).__init__()
        # Use a pre-trained MobileNetV2 as the backbone
        self.backbone = models.mobilenet_v2(pretrained=True)
        
        # Replace the final classifier layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)