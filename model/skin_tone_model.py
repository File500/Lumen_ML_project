import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b3, EfficientNet_B3_Weights, efficientnet_b5, EfficientNet_B5_Weights

class EfficientNetB0SkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super(EfficientNetB0SkinToneClassifier, self).__init__()
        
        # Load pre-trained EfficientNet
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Freeze feature extractor if requested
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get input features dimension (1280 for EfficientNet-B0)
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Create feature extractor part (what you want to save and reuse)
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Create final classifier part (what you'll discard for transfer learning)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Get backbone features
        x = self.backbone(x)
        # Apply feature extractor to get 512-dim features
        features = self.feature_extractor(x)
        # Apply classifier to get final output
        return self.classifier(features)
    
    def extract_features(self, x):
        """Extract the 512-dimensional features"""
        x = self.backbone(x)
        return self.feature_extractor(x)
    

class EfficientNetB3SkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super(EfficientNetB3SkinToneClassifier, self).__init__()
        
        # Load pre-trained EfficientNet B3
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Freeze feature extractor if requested
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get input features dimension (1536 for EfficientNet-B3)
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Create feature extractor part (what you want to save and reuse)
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 768),  # Adjusted for larger input features
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Create final classifier part (what you'll discard for transfer learning)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Get backbone features
        x = self.backbone(x)
        # Apply feature extractor to get features
        features = self.feature_extractor(x)
        # Apply classifier to get final output
        return self.classifier(features)
    
    def extract_features(self, x):
        """Extract the 128-dimensional features"""
        x = self.backbone(x)
        return self.feature_extractor(x)
    
class EfficientNetB5SkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10, freeze_features=True):
        super(EfficientNetB5SkinToneClassifier, self).__init__()
        
        # Load pre-trained EfficientNet B5
        self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        
        # Freeze feature extractor if requested
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Get input features dimension (2048 for EfficientNet-B5)
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Create feature extractor part (what you want to save and reuse)
        self.feature_extractor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1024),  # Adjusted for larger input features
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),  # Added an extra layer for more capacity
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Create final classifier part (what you'll discard for transfer learning)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Get backbone features
        x = self.backbone(x)
        # Apply feature extractor to get features
        features = self.feature_extractor(x)
        # Apply classifier to get final output
        return self.classifier(features)
    
    def extract_features(self, x):
        """Extract the 128-dimensional features"""
        x = self.backbone(x)
        return self.feature_extractor(x)