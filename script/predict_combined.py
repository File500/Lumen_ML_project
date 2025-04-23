import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import warnings

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import EfficientNetB3SkinToneClassifier
from model.modified_melanoma_model import ModifiedMelanomaClassifier
from model.combined_model import CombinedTransferModel

warnings.filterwarnings("ignore", category=UserWarning)

# Set fixed device - only call this once
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 
                      "cuda:0" if torch.cuda.is_available() else "cpu")

CPU_DEVICE = torch.device("cpu")

# Helper function to recursively convert all tensors to float
def convert_tensors_to_float(obj):
    if isinstance(obj, torch.Tensor):
        return obj.float()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_float(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_tensors_to_float(v) for v in obj)
    return obj

# Image cleaning function
def mask_dark_pixels_torch(img, threshold=70):
    """Clean image by masking dark pixels (typically dark borders/artifacts)"""
    if hasattr(img, 'convert'):
        img = img.convert('RGB')

    img_tensor = transforms.ToTensor()(img).to(DEVICE)
    gray_tensor = TF.rgb_to_grayscale(img_tensor)
    threshold_norm = threshold / 255.0
    dark_mask = (gray_tensor < threshold_norm).float()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=DEVICE).view(1, 1, 3, 3).float()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=DEVICE).view(1, 1, 3, 3).float()

    padded_gray = F.pad(gray_tensor.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    edge_x = F.conv2d(padded_gray, sobel_x)
    edge_y = F.conv2d(padded_gray, sobel_y)
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(0)
    edge_mask = (edges > 0.1).float()

    combined_mask = torch.clamp(dark_mask + edge_mask, 0, 1)

    kernel = torch.ones((1, 1, 5, 5), device=DEVICE).float()
    dilated_mask = F.conv2d(combined_mask.unsqueeze(0), kernel, padding=2).squeeze(0)
    dilated_mask = (dilated_mask > 0).float()

    close_kernel = torch.ones((1, 1, 7, 7), device=DEVICE).float()
    closing_mask = F.conv2d(dilated_mask.unsqueeze(0), close_kernel, padding=3)
    closing_mask = F.conv_transpose2d((closing_mask > 0).float(), close_kernel, padding=3).squeeze(0)
    closing_mask = (closing_mask > 30).float()

    mask_3d = closing_mask.expand_as(img_tensor)
    blurred = TF.gaussian_blur(img_tensor, kernel_size=[9, 9], sigma=[4.0, 4.0])
    inpainted = img_tensor * (1 - mask_3d) + blurred * mask_3d

    return TF.to_pil_image(inpainted.cpu())

# Function to preprocess single image (cleaned and resized)
def preprocess_image(img, target_size=(300, 300)):
    """Preprocess image for model input - always includes cleaning"""
    # Clean the image
    cleaned_img = mask_dark_pixels_torch(img, threshold=70)
    
    # Resize and center crop
    transform = transforms.Compose([
        transforms.Resize(min(target_size)),
        transforms.CenterCrop(min(cleaned_img.size)),
        transforms.Pad(padding=[
            (target_size[0] - cleaned_img.width) // 2 if cleaned_img.width < target_size[0] else 0,
            (target_size[1] - cleaned_img.height) // 2 if cleaned_img.height < target_size[1] else 0
        ]),
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    
    return transform(cleaned_img)

# Create a dataset that applies cleaning to each image
class CleanImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_files, target_size=(300, 300)):
        self.img_dir = img_dir
        self.img_files = img_files
        self.target_size = target_size
        
    def __len__(self):
        return len(self.img_files)
        
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            # Apply cleaning and preprocessing
            processed_image = preprocess_image(image, self.target_size)
            return processed_image, img_name
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            # Return a blank image in case of error
            return torch.zeros(3, self.target_size[0], self.target_size[1]), img_name

# Custom module to ensure all parameters and buffers are float
def convert_module_to_float(module):
    """Recursively converts all parameters and buffers in module to float type"""
    for param in module.parameters():
        param.data = param.data.float()
    
    for buffer in module.buffers():
        if buffer.data.dtype != torch.float32:
            buffer.data = buffer.data.float()
    
    for child in module.children():
        convert_module_to_float(child)
    
    return module

def load_combined_model(skin_model_path, melanoma_model_path, combined_model_path):
    """Load all models needed for combined prediction with careful type handling"""
    # 1. Create and load skin tone model
    print(f"Loading skin tone model from {skin_model_path}")
    skin_model = EfficientNetB3SkinToneClassifier(num_classes=7)
    
    # Load state dict with careful type handling
    skin_state_dict = torch.load(skin_model_path, map_location=DEVICE)
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
    
    # Convert all tensors to float before loading
    state_dict = convert_tensors_to_float(state_dict)
    skin_model.load_state_dict(state_dict, strict=False)
    skin_model = convert_module_to_float(skin_model)
    skin_model.to(DEVICE)
    skin_model.eval()
    
    # 2. Create and load melanoma model
    print(f"Loading melanoma model from {melanoma_model_path}")
    melanoma_model = ModifiedMelanomaClassifier(num_classes=2, binary_mode=True)
    
    # Load state dict with careful type handling
    melanoma_state_dict = torch.load(melanoma_model_path, map_location=DEVICE)
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
    
    # Convert all tensors to float before loading
    state_dict = convert_tensors_to_float(state_dict)
    melanoma_model.load_state_dict(state_dict, strict=False)
    melanoma_model = convert_module_to_float(melanoma_model)
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
    
    # Force all parameters to float
    combined_model = convert_module_to_float(combined_model)
    combined_model.to(DEVICE)
    
    # Load combined model weights
    print(f"Loading combined model weights from {combined_model_path}")
    combined_state_dict = torch.load(combined_model_path, map_location=DEVICE)
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
    
    # Convert all tensors to float before loading
    state_dict = convert_tensors_to_float(state_dict)
    combined_model.load_state_dict(state_dict, strict=False)
    combined_model = convert_module_to_float(combined_model)
    combined_model.eval()
    
    print("Model loading complete")
    return combined_model

def predict_images_in_directory(model, img_dir, output_csv, threshold=0.3, batch_size=32, num_workers=4):
    """
    Run predictions on all images in a directory - always cleans images
    
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
    
    # Create dataset and dataloader with cleaning built-in
    dataset = CleanImageDataset(img_dir, img_files, target_size=(300, 300))

        
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Process batches
    results = []
    
    print(f"Processing {len(img_files)} images in batches of {batch_size}")
    
    for batch_images, batch_names in tqdm(dataloader, desc="Predicting"):
        # Move batch to device
        batch_images = batch_images.to(DEVICE).float()
        
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
    # Print device info once at the beginning
    print(f"Using device: {DEVICE}" + 
          (f" ({torch.cuda.get_device_name(DEVICE.index)})" if DEVICE.type == 'cuda' else ""))
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_PATH = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(PROJECT_PATH, "data")
    
    # Settings
    img_dir = os.path.join(DATA_PATH, "ISIC_2020_Test_Input")
    output_csv = os.path.join(DATA_PATH, "Test_predictions.csv")
    threshold = 0.3
    batch_size = 64  
    
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