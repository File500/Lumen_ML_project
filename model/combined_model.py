import torch
import torch.nn as nn
import torchvision.models as models


# Combined Transfer Learning Model
class CombinedTransferModel(nn.Module):
    def __init__(self, skin_tone_model, melanoma_model, num_classes=2, binary_mode=True):
        super(CombinedTransferModel, self).__init__()
        
        # Store the pretrained models
        self.skin_tone_model = skin_tone_model
        self.melanoma_model = melanoma_model
        
        # Freeze the parameters of the pretrained models
        for param in self.skin_tone_model.parameters():
            param.requires_grad = False
        
        for param in self.melanoma_model.parameters():
            param.requires_grad = True
        
        # Combined feature dimension (256 from skin model + 128 from melanoma model)
        combined_dim = 256 + 128
        
        # For binary classification with BCEWithLogitsLoss, use 1 output
        # For multi-class classification with CrossEntropyLoss, use num_classes outputs
        output_dim = 1 if binary_mode else num_classes
        
        # New classifier for combined features
        self.combined_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.4),  # Increase dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )
        
        self.binary_mode = binary_mode
    
    def forward(self, x):
        # Extract features from both models
        with torch.no_grad():  # Ensure skin tone model remains frozen
            skin_features = self.skin_tone_model.get_features(x)
        
        # Extract features from melanoma model (can be fine-tuned)
        melanoma_features = self.melanoma_model.get_features(x)
        
        # Scale features to have similar magnitudes (very important!)
        skin_features = skin_features / (torch.norm(skin_features, dim=1, keepdim=True) + 1e-8)
        melanoma_features = melanoma_features / (torch.norm(melanoma_features, dim=1, keepdim=True) + 1e-8)
        
        # Concatenate features with relative importance weighting
        # Give melanoma features slightly more weight 
        combined_features = torch.cat((skin_features * 0.8, melanoma_features * 1.2), dim=1)
        
        # Pass through the combined classifier
        output = self.combined_classifier(combined_features)
        return output