import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedMelanomaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PretrainedMelanomaClassifier, self).__init__()
        
        # Load EfficientNet-B0 with pretrained weights
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Modify the classifier to fit the number of classes
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

# Example usage
model = PretrainedMelanomaClassifier(num_classes=2)
print(model)
