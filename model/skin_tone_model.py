import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b3, EfficientNet_B3_Weights, efficientnet_b5, EfficientNet_B5_Weights, vgg16, VGG16_Weights

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
            nn.Dropout(0.2),  
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),  
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        
        # Classifier with ordinal awareness
        self.classifier = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(256, num_classes)
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
    

class AttentionSkinToneClassifier(nn.Module):
    def __init__(self, num_classes=7, model_type='b3', freeze_features=False):
        super().__init__()
        
        # Load appropriate backbone based on model type
        if model_type == 'b0':
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = 1280
        elif model_type == 'b3':
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            in_features = 1536
        elif model_type == 'b5':
            self.backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
            in_features = 2048
        
        # Freeze feature extractor if requested
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier with identity
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism to focus on skin regions
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LayerNorm(1024),  # Layer norm instead of batch norm
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Classifier with ordinal awareness
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention
        attention_mask = self.attention(features)
        features = features * attention_mask
        
        # Global pooling and classification
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.feature_extractor(features)
        return self.classifier(features)
    
    def get_features(self, x):
        """
        Extract features from the model before classification
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Extracted features
        """
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention
        attention_mask = self.attention(features)
        features = features * attention_mask
        
        # Global pooling
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Feature extraction
        features = self.feature_extractor(features)
        
        return features
    

class SkinToneFusionModel(nn.Module):
    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()
       
        # VGG-16 backbone with weights
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Selective freezing - keep last few layers trainable
        if freeze_backbone:
            # Freeze only early layers (first 70%)
            layers_to_freeze = int(len(list(self.vgg.features)) * 0.7)
            for i, param in enumerate(self.vgg.features.parameters()):
                if i < layers_to_freeze:
                    param.requires_grad = False
        
        self.vgg_features = self.vgg.features
        
        # Enhanced custom CNN for YCbCr with bottleneck/residual connections
        self.custom_cnn = nn.Sequential(
            # First block with residual connection
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block with attention
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Attention mechanism for skin tone regions
            ChannelAttention(128),
            SpatialAttention(),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Global pooling
        self.vgg_gap = nn.AdaptiveAvgPool2d(1)
        self.custom_gap = nn.AdaptiveAvgPool2d(1)
        
        # Class-balanced fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # Enhanced classifier with class-specific attention
        self.classifier = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # VGG path
        vgg_features = self.vgg_features(x)
        vgg_features = self.vgg_gap(vgg_features).view(batch_size, -1)
        
        # YCbCr path
        ycbcr = rgb_to_ycbcr(x)
        custom_features = self.custom_cnn(ycbcr)
        custom_features = self.custom_gap(custom_features).view(batch_size, -1)
        
        # Fusion
        combined = torch.cat([vgg_features, custom_features], dim=1)
        fused = self.fusion(combined)
        
        # Classification
        output = self.classifier(fused)
        
        return output

# Attention modules for focusing on skin regions
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

def rgb_to_ycbcr(image):
    """
    Convert RGB tensor to YCbCr color space
    """
    # Create transformation matrix
    transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.500],
        [0.500, -0.419, -0.081]
    ], dtype=torch.float32).to(image.device)
    
    # Ensure image is in right format [B, C, H, W]
    B, C, H, W = image.shape
    
    # Reshape to [B, H, W, C]
    image = image.permute(0, 2, 3, 1)
    
    # Apply transformation
    result = torch.zeros_like(image)
    
    # Y channel
    result[..., 0] = torch.matmul(image, transform_matrix[0])
    
    # Cb channel (add 128)
    result[..., 1] = torch.matmul(image, transform_matrix[1]) + 0.5
    
    # Cr channel (add 128)
    result[..., 2] = torch.matmul(image, transform_matrix[2]) + 0.5
    
    # Return in [B, C, H, W] format
    return result.permute(0, 3, 1, 2)