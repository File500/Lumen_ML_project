import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedMelanomaClassifier(nn.Module):
    def __init__(self, num_classes=2, binary_mode=True):
        super(PretrainedMelanomaClassifier, self).__init__()
        
        # Load EfficientNet-B0 with pretrained weights
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # For binary classification with BCEWithLogitsLoss, use 1 output
        # For multi-class classification with CrossEntropyLoss, use num_classes outputs
        output_dim = 1 if binary_mode else num_classes
        
        # Modify the classifier to fit the number of classes
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536, output_dim) 
            #1280 for b0 model 
            #1536 for b3 model
            #2048 for b5 model
        )
        
        self.binary_mode = binary_mode
    
    def forward(self, x):
        return self.efficientnet(x)