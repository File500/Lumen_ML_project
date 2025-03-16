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
    

class EfficientNetSkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10, efficientnet_version='b0'):
        super(EfficientNetSkinToneClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        
        # Replace the classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Additional model variants you might want to explore
class ResNetSkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10, resnet_version='resnet50'):
        super(ResNetSkinToneClassifier, self).__init__()
        
        # Load pre-trained ResNet
        self.backbone = models.resnet50(pretrained=True)
       
        
        # Modify the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)