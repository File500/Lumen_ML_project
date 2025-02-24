import torch
import torch.nn as nn
import torch.nn.functional as F

#EfficientNet-B0 architecture

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, kernel_size, stride):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1
        
        self.conv = nn.Sequential(
            # Expansion layer
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            
            # Depthwise Convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            
            # Squeeze-and-Excitation
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
            
            # Pointwise Convolution
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return out + x  # Skip connection
        return out


class MelanomaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MelanomaClassifier, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.blocks = nn.Sequential(
            MBConv(32, 16, expand_ratio=1, kernel_size=3, stride=1),  # 320x240
            MBConv(16, 24, expand_ratio=6, kernel_size=3, stride=2),  # 160x120
            MBConv(24, 40, expand_ratio=6, kernel_size=5, stride=2),  # 80x60
            MBConv(40, 80, expand_ratio=6, kernel_size=3, stride=2),  # 40x30
            MBConv(80, 112, expand_ratio=6, kernel_size=5, stride=1), # 40x30
            MBConv(112, 192, expand_ratio=6, kernel_size=5, stride=2),# 20x15
            MBConv(192, 320, expand_ratio=6, kernel_size=3, stride=1) # 20x15
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),  # 20x15 -> 20x15
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1)  # Reduce to 1x1
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )



    def forward(self, x):
        x = self.stem(x)      # Process input
        x = self.blocks(x)    # EfficientNet MBConv blocks
        x = self.head(x)      # Feature extraction
        x = self.classifier(x)  # Classification
        return x
