import torch
import torch.nn as nn
import torchvision.models as models

# Modified Melanoma Classifier with 128-neuron layer before output
class ModifiedMelanomaClassifier(nn.Module):
    def __init__(self, num_classes=2, binary_mode=True):
        super(ModifiedMelanomaClassifier, self).__init__()
       
        # Load EfficientNet-B3 with pretrained weights
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Get the input features size for the final classifier
        in_features = 1536  # for EfficientNet-B3
        
        # Add a 128-neuron layer before the output
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1 if binary_mode else num_classes)
        )
        
        self.binary_mode = binary_mode
   
    def forward(self, x):
        return self.efficientnet(x)
    
    # Method to get the 128-dimensional features before the final classification layer
    def get_features(self, x):
        # Extract features from backbone
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Get output from the first part of the classifier (128-dim features)
        features = self.efficientnet.classifier[0](x)  # Dropout
        features = self.efficientnet.classifier[1](features)  # Linear to 128
        features = self.efficientnet.classifier[2](features)  # ReLU
        
        return features
