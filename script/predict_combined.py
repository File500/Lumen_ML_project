import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import EfficientNetB3SkinToneClassifier
from model.modified_melanoma_model import ModifiedMelanomaClassifier
from model.combined_model import CombinedTransferModel

# Set fixed device - only call this once
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 
                      "cuda:0" if torch.cuda.is_available() else "cpu")

# Define dataset class outside the function to avoid pickle errors with multiprocessing
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_files, transform):
        self.img_dir = img_dir
        self.img_files = img_files
        self.transform = transform
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, img_name
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            # Return a blank image in case of error
            return torch.zeros(3, 300, 300), img_name

def load_combined_model(skin_model_path, melanoma_model_path, combined_model_path):
    """Load all models needed for combined prediction"""
    # 1. Load skin tone classifier
    print(f"Loading skin tone model from {skin_model_path}")
    skin_model = EfficientNetB3SkinToneClassifier(num_classes=7)
    skin_state_dict = torch.load(skin_model_path, map_location=DEVICE)
    
    # Check and load correct format for skin model
    if isinstance(skin_state_dict, dict):
        if 'model_state' in skin_state_dict:
            state_dict = skin_state_dict['model_state']
        elif 'state_dict' in skin_state_dict:
            state_dict = skin_state_dict['state_dict']
        elif 'model_state_dict' in skin_state_dict:
            state_dict = skin_state_dict['model_state_dict']
        else:
            state_dict = skin_state_dict
    else:
        state_dict = skin_state_dict
    
    skin_model.load_state_dict(state_dict, strict=False)
    skin_model.to(DEVICE)
    skin_model.eval()
    
    # 2. Load melanoma model
    print(f"Loading melanoma model from {melanoma_model_path}")
    melanoma_model = ModifiedMelanomaClassifier(num_classes=2, binary_mode=True)
    melanoma_state_dict = torch.load(melanoma_model_path, map_location=DEVICE)
    
    # Check and load correct format for melanoma model
    if isinstance(melanoma_state_dict, dict):
        if 'model_state' in melanoma_state_dict:
            state_dict = melanoma_state_dict['model_state']
        elif 'state_dict' in melanoma_state_dict:
            state_dict = melanoma_state_dict['state_dict']
        elif 'model_state_dict' in melanoma_state_dict:
            state_dict = melanoma_state_dict['model_state_dict']
        else:
            state_dict = melanoma_state_dict
    else:
        state_dict = melanoma_state_dict
    
    melanoma_model.load_state_dict(state_dict, strict=False)
    melanoma_model.to(DEVICE)
    melanoma_model.eval()
    
    # 3. Create and load combined model
    print(f"Creating combined model...")
    combined_model = CombinedTransferModel(
        skin_tone_model=skin_model,
        melanoma_model=melanoma_model,
        num_classes=2,
        binary_mode=True
    )
    
    # Load combined model weights
    print(f"Loading combined model weights from {combined_model_path}")
    combined_state_dict = torch.load(combined_model_path, map_location=DEVICE)
    
    # Check and load correct format for combined model
    if isinstance(combined_state_dict, dict):
        if 'model_state' in combined_state_dict:
            state_dict = combined_state_dict['model_state']
        elif 'state_dict' in combined_state_dict:
            state_dict = combined_state_dict['state_dict']
        elif 'model_state_dict' in combined_state_dict:
            state_dict = combined_state_dict['model_state_dict']
        else:
            state_dict = combined_state_dict
    else:
        state_dict = combined_state_dict
    
    combined_model.load_state_dict(state_dict, strict=False)
    combined_model.to(DEVICE)
    combined_model.eval()
    
    print("Model loading complete")
    return combined_model

def predict_images_in_directory(model, img_dir, output_csv, threshold=0.3, batch_size=32, num_workers=4):
    """
    Run predictions on all images in a directory - optimized for speed
    
    Args:
        model: The loaded combined model
        img_dir: Directory containing images to predict
        output_csv: Path to save predictions CSV
        threshold: Probability threshold for binary classification
        batch_size: Batch size for processing
        num_workers: Number of worker processes for data loading
    """
    model.eval()
    torch.set_grad_enabled(False)  # Disable gradient computation for the entire process
    
    # Get all image files
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(img_files)} images in {img_dir}")
    
    # Define a simple transform
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = SimpleImageDataset(img_dir, img_files, transform)
    
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    
    # Process batches
    results = []
    
    print(f"Processing {len(img_files)} images in batches of {batch_size}...")
    
    for batch_images, batch_names in tqdm(dataloader, desc="Predicting"):
        # Move batch to device
        batch_images = batch_images.to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(batch_images)
            probs = torch.sigmoid(outputs)
            
            # Process each image in the batch
            batch_probs = probs.cpu().numpy()
            batch_preds = (batch_probs > threshold).astype(int)
            
            for i in range(len(batch_names)):
                results.append({
                    'image_name': batch_names[i],
                    'prediction': int(batch_preds[i][0]),
                    'probability': float(batch_probs[i][0])
                })
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    
    # Print summary
    malignant_count = sum(results_df['prediction'] == 1)
    benign_count = sum(results_df['prediction'] == 0)
    print("\n===== Prediction Summary =====")
    print(f"Total images processed: {len(results_df)}")
    print(f"Benign predictions (0): {benign_count} ({benign_count/len(results_df)*100:.1f}%)")
    print(f"Malignant predictions (1): {malignant_count} ({malignant_count/len(results_df)*100:.1f}%)")
    print(f"Results saved to: {output_csv}")
    
    return results_df

def main():
    """Main function to run predictions"""
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_PATH = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    
    # Settings
    img_dir = os.path.join(DATA_PATH, "ISIC_2020_Test_Input")
    output_csv = os.path.join(DATA_PATH, "Test_predictions.csv")
    threshold = 0.3
    batch_size = 64  # Adjust based on your GPU memory
    
    # Model paths
    config = {
        "skin_model_path": os.path.join(PROJECT_PATH, "trained_model", "skin_type_classifier", "EFNet_b3_300X300_final", "final_model.pth"),
        "melanoma_model_path": os.path.join(PROJECT_PATH, "trained_model", "feature_extractor_melanoma_classifier_final", "modified_melanoma_model.pth"), 
        "combined_model_path": os.path.join(PROJECT_PATH, "trained_model", "combined_model_final", "best_combined_model.pth"),
        "threshold": threshold
    }
    
    # Check if files exist
    if not os.path.exists(img_dir):
        print(f"ERROR: Image directory not found at {img_dir}")
        return
    
    for key in ["skin_model_path", "melanoma_model_path", "combined_model_path"]:
        if not os.path.exists(config[key]):
            print(f"ERROR: {key} not found at {config[key]}")
            return
    
    # Load combined model (only once)
    model = load_combined_model(
        skin_model_path=config["skin_model_path"],
        melanoma_model_path=config["melanoma_model_path"],
        combined_model_path=config["combined_model_path"]
    )
    
    # Run predictions
    results_df = predict_images_in_directory(
        model=model,
        img_dir=img_dir,
        output_csv=output_csv,
        threshold=config["threshold"],
        batch_size=batch_size
    )

if __name__ == "__main__":
    main()