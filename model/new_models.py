import torch
import torch.nn as nn
import torchvision.models as models

# Modified Melanoma Classifier with 128 neurons before output
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
        # This extracts features up through the ReLU activation after the 128-neuron layer
        features = self.efficientnet.classifier[0](x)  # Dropout
        features = self.efficientnet.classifier[1](features)  # Linear to 128
        features = self.efficientnet.classifier[2](features)  # ReLU
        
        return features

# Function to load pretrained weights and transfer them to the modified model
def load_and_transfer_weights(original_model_path, modified_model):
    """
    Load weights from an original model and transfer compatible weights to the modified model
    
    Parameters:
    - original_model_path: Path to the .pth file containing the original model weights
    - modified_model: The new model with the 128-neuron layer
    
    Returns:
    - The modified model with transferred weights
    """
    # Load the original state dict
    original_state_dict = torch.load(original_model_path)
    
    # Create a new state dict for the modified model
    modified_state_dict = modified_model.state_dict()
    
    # Transfer weights for all layers except the classifier
    for name, param in original_state_dict.items():
        # Skip classifier weights as they won't be compatible with our new structure
        if 'classifier' not in name:
            modified_state_dict[name] = param
    
    # Load the modified state dict into the model
    modified_model.load_state_dict(modified_state_dict, strict=False)
    
    return modified_model


# Combined Transfer Learning Model (using the 128-feature layer from melanoma model)
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
            param.requires_grad = False
        
        # Combined feature dimension (256 from skin model + 128 from melanoma model)
        combined_dim = 256 + 128
        
        # For binary classification with BCEWithLogitsLoss, use 1 output
        # For multi-class classification with CrossEntropyLoss, use num_classes outputs
        output_dim = 1 if binary_mode else num_classes
        
        # New classifier for combined features
        self.combined_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
        self.binary_mode = binary_mode
    
    def forward(self, x):
        # Extract features from both models
        skin_features = self.skin_tone_model.get_features(x)  # 256-dim features
        melanoma_features = self.melanoma_model.get_features(x)  # 128-dim features
        
        # Concatenate features
        combined_features = torch.cat((skin_features, melanoma_features), dim=1)  # 384-dim
        
        # Pass through the combined classifier
        output = self.combined_classifier(combined_features)
        return output


# Example usage
def example():
    # Create the modified model
    modified_model = ModifiedMelanomaClassifier(num_classes=2, binary_mode=True)
    
    # Load weights from the original model (if available)
    original_model_path = 'path_to_original_melanoma_model.pth'
    modified_model = load_and_transfer_weights(original_model_path, modified_model)
    
    # Now you can use the modified model with 128-feature layer
    # For example, to get the 128-dim features:
    example_input = torch.randn(1, 3, 224, 224)
    features = modified_model.get_features(example_input)
    print(f"Feature shape: {features.shape}")  # Should be [1, 128]
    
    return modified_model