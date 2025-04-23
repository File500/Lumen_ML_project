import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import collections
import traceback

# Add the parent directory to the path so we can import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the model - if it fails, provide helpful error message
try:
    from model.skin_tone_model import EfficientNetB3SkinToneClassifier
except ImportError:
    print("ERROR: Could not import model. Make sure your model directory is in the Python path.")
    print("You may need to set: export PYTHONPATH=$PYTHONPATH:/path/to/your/project/root")
    traceback.print_exc()

class MonkScaleDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, cache_size=500):
        """Simple dataset class for prediction only (no labels required)"""
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.cache_size = cache_size
        self.cache = collections.OrderedDict()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image name and add .jpg extension if needed
        image_name = self.df.iloc[idx]['image_name']
        if not image_name.endswith('.jpg'):
            image_name = f"{image_name}.jpg"
        
        img_path = os.path.join(self.image_dir, image_name)
        
        # Check cache first
        if img_path in self.cache:
            image = self.cache.pop(img_path)
            self.cache[img_path] = image
        else:
            try:
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Add to cache if enabled
                if self.cache_size > 0:
                    if len(self.cache) >= self.cache_size:
                        self.cache.popitem(last=False)  # Remove oldest item
                    self.cache[img_path] = image
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a black image if there's an error
                image = Image.new('RGB', (300, 300), (0, 0, 0))
        
        # Apply transformations
        if self.transform:
            transformed_image = self.transform(image)
        else:
            # Convert to tensor if no transform provided
            from torchvision import transforms
            transformer = transforms.ToTensor()
            transformed_image = transformer(image)
        
        return transformed_image, idx

def predict_monk_skin_types(model_path, csv_path, image_dir, output_csv_path=None, batch_size=32):
    """
    Run prediction on all images in the CSV and add a monk_skin_type column
    
    Args:
        model_path: Path to the saved model
        csv_path: Path to the CSV file containing image_name column
        image_dir: Directory containing the images
        output_csv_path: Where to save the CSV with predictions (default: adds _predictions suffix)
        batch_size: Batch size for processing
    """
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Set default output path if not provided
        if output_csv_path is None:
            base_path = os.path.splitext(csv_path)[0]
            output_csv_path = f"{base_path}_predictions.csv"
        
        # Load the CSV
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} images to process")
        
        # Define transform for inference
        val_transform = transforms.Compose([
            transforms.Resize((300, 300)),  # Adjust size as needed for your model
            transforms.ToTensor(),
        ])
        
        # Create dataset and dataloader
        dataset = MonkScaleDataset(df, image_dir, transform=val_transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,  # Important: keep order the same as CSV
            num_workers=4,
            pin_memory=True
        )
        
        # Load the model
        print(f"Loading model from {model_path}")

        # Try loading the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        print("Checkpoint loaded successfully")

        # Initialize the model
        model = EfficientNetB3SkinToneClassifier(num_classes=7)

        # Handle dictionary checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Standard checkpoint format with model_state_dict key
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded from checkpoint's model_state_dict")
            elif 'state_dict' in checkpoint:
                # Alternative checkpoint format
                model.load_state_dict(checkpoint['state_dict'])
                print("Model loaded from checkpoint's state_dict")
            else:
                # Try loading directly if the dict itself is a state dict
                model.load_state_dict(checkpoint)
                print("Model loaded directly from checkpoint dictionary")
        else:
            # If it's not a dictionary, assume it's the model itself
            model = checkpoint
            print("Full model loaded directly")

        model = model.to(device)
        model.eval()
        print("Model prepared for inference")
        
        # Make predictions
        print("Running predictions...")
        predictions = {}  # Map of index -> prediction
        
        with torch.no_grad():
            for inputs, indices in tqdm(dataloader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Convert from 0-indexed to 1-indexed (Monk Scale 1-7)
                for idx, pred in zip(indices.cpu().numpy(), preds.cpu().numpy()):
                    predictions[int(idx)] = int(pred) + 1
        
        # Add predictions to dataframe
        df['monk_skin_type'] = [predictions.get(i, None) for i in range(len(df))]
        
        # Save results
        df.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to: {output_csv_path}")
        
        # Display prediction statistics
        print("\nPrediction Distribution:")
        pred_counts = df['monk_skin_type'].value_counts().sort_index()
        for scale, count in pred_counts.items():
            percentage = (count / len(df)) * 100
            print(f"Monk Scale {scale}: {count} predictions ({percentage:.2f}%)")
            
        return df, True, "Prediction completed successfully!"
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return None, False, f"Error: {str(e)}"

# SETTINGS
# ===========================================
# 
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_PATH = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(PROJECT_PATH, "data")

MODEL_PATH = os.path.join(PROJECT_PATH, "trained_model", "skin_type_classifier", "EFNet_b3_300X300_final", "final_model.pth")

# Path to your CSV file with image_name column
CSV_PATH = os.path.join(DATA_PATH, "deduplicated_monk_scale_dataset.csv")

# Directory containing the images
IMAGE_DIR = os.path.join(DATA_PATH, "train_300X300_processed") 

# Where to save the predictions (leave as None to auto-generate the name)
OUTPUT_PATH = None

# Batch size for processing (reduce if you have memory issues)
BATCH_SIZE = 32
# ===========================================

if __name__ == "__main__":
    print("=" * 70)
    print("MONK SKIN TYPE PREDICTION SCRIPT")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"CSV: {CSV_PATH}")
    print(f"Image directory: {IMAGE_DIR}")
    print(f"Output: {'Auto-generated' if OUTPUT_PATH is None else OUTPUT_PATH}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 70)
    print("Starting prediction process...")
    
    # Run prediction with predefined settings
    df, success, message = predict_monk_skin_types(
        model_path=MODEL_PATH,
        csv_path=CSV_PATH,
        image_dir=IMAGE_DIR,
        output_csv_path=OUTPUT_PATH,
        batch_size=BATCH_SIZE
    )
    
    if success:
        print("\nProcess completed successfully!")
        print("You can find your results in the output CSV file.")
    else:
        print("\nProcess failed!")
        print(message)
    
    print("\nPress Enter to exit...")
    input()  # This keeps the console window open after completion