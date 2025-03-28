import os
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, precision_recall_fscore_support
from PIL import Image
import copy
import time
import collections
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import cv2

# Import the model we created earlier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import EfficientNetB0SkinToneClassifier, EfficientNetB3SkinToneClassifier, EfficientNetB5SkinToneClassifier
from ETL_and_EDA.skin_tone.feature_extraction import create_skin_mask, crop_black_borders

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class BaseConfig:
    # Common data paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(CURRENT_DIR)
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    CSV_PATH = os.path.join(DATA_DIR, "monk_scale_dataset.csv")
    
    # Common settings
    APPLY_MASK = True
    MASK_TYPE = 'binary'  # 'binary', 'focus', or 'highlight'
    NUM_CLASSES = 7
    FREEZE_FEATURES = False
    
    # Data split
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Device settings
    GPU_ID = 2
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    # Common training parameters
    BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001  # Reduced for fine-tuning
    WEIGHT_DECAY = 1e-5
    PATIENCE = 10

    # Transformation paramethers
    RANDOMROTATION = 10
    BRIGHTNESS = 0.1
    CONTRAST = 0.1
    SATURATION = 0.1
    HUE = 0.1
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


# EfficientNet B0 Config (224x224)
class ConfigB0(BaseConfig):
    MODEL_TYPE = 'b0'
    IMAGE_DIMENSION = 224
    
    # Specific directories for B0
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}")
    PREPROCESS_DIR = os.path.join(BaseConfig.DATA_DIR, f"masked_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}", BaseConfig.MASK_TYPE)
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{BaseConfig.MASK_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}")


# EfficientNet B3 Config (300x300)
class ConfigB3(BaseConfig):
    MODEL_TYPE = 'b3'
    IMAGE_DIMENSION = 300
    
    # Specific directories for B3
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}")
    PREPROCESS_DIR = os.path.join(BaseConfig.DATA_DIR, f"masked_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}", BaseConfig.MASK_TYPE)
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{BaseConfig.MASK_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}")
    
    # Slightly different training params (optional)
    BATCH_SIZE = 48  # Reduced batch size for larger model


# EfficientNet B5 Config (456x456)
class ConfigB5(BaseConfig):
    MODEL_TYPE = 'b5'
    IMAGE_DIMENSION = 456
    
    # Specific directories for B5
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}")
    PREPROCESS_DIR = os.path.join(BaseConfig.DATA_DIR, f"masked_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}", BaseConfig.MASK_TYPE) 
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{BaseConfig.MASK_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}")
    
    # Adjusted parameters for the larger model
    BATCH_SIZE = 32  # Smaller batch size
    LEARNING_RATE = 0.00005  # Even smaller learning rate

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
    
    def __call__(self, val_loss):
        """
        Args:
            val_loss (float): current validation loss
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Create the dataset class
class MaskedMonkScaleDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, cache_size=5000, 
                 apply_mask=True, mask_type='binary', mask_background=(0,0,0),
                 preprocess_dir=None):
        """
        Dataset for Monk Scale with masked images.
        
        Args:
            df (pandas.DataFrame): DataFrame with image_name and monk_scale_type
            image_dir (string): Directory with all the images
            transform (callable, optional): Transform to be applied on the masked images
            cache_size (int): Maximum size of the LRU cache
            apply_mask (bool): Whether to apply skin masking
            mask_type (str): Type of masking to apply:
                - 'binary': Black out non-skin regions
                - 'focus': Keep only skin regions (crop to skin)
                - 'highlight': Keep original image but highlight skin regions
            mask_background (tuple): RGB values for masked regions (for binary mask)
            preprocess_dir (str, optional): Directory to save preprocessed masked images
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.cache_size = cache_size
        self.apply_mask = apply_mask
        self.mask_type = mask_type
        self.mask_background = mask_background
        self.preprocess_dir = preprocess_dir
        
        # Create preprocessing directory if specified
        if self.preprocess_dir:
            os.makedirs(self.preprocess_dir, exist_ok=True)
        
        # Initialize LRU cache
        self.cache = collections.OrderedDict()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image name and add .jpg extension if it doesn't have it
        image_name = self.df.iloc[idx]['image_name']
        if not image_name.endswith('.jpg'):
            image_name = f"{image_name}.jpg"
        
        img_path = os.path.join(self.image_dir, image_name)
        
        # Check if preprocessed image exists
        if self.preprocess_dir and self.apply_mask:
            preprocessed_path = os.path.join(self.preprocess_dir, image_name)
            if os.path.exists(preprocessed_path):
                # Load preprocessed image
                image = Image.open(preprocessed_path).convert('RGB')
            else:
                # Process and save
                image = self._process_and_save_image(img_path, preprocessed_path)
        else:
            # Check if image is in cache
            if img_path in self.cache:
                # Get image from cache and mark as recently used by moving to end
                image = self.cache.pop(img_path)
                self.cache[img_path] = image
            else:
                # Process image
                image = self._process_image(img_path)
                
                # Add to cache if caching is enabled
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)  # Remove least recently used
                self.cache[img_path] = image
        
        # Apply transformations
        if self.transform:
            # Create a copy before transforming to keep original in cache
            transformed_image = self.transform(image.copy())
        else:
            transformed_image = image
        
        # Get label
        label = self.df.iloc[idx]['monk_scale_type'] - 1
        
        return transformed_image, label
    
    def _process_image(self, img_path):
        """Process a single image by applying masking if needed."""
        if not self.apply_mask:
            # Just load the image normally if no masking
            return Image.open(img_path).convert('RGB')
        
        # Read the image with OpenCV for processing
        img = cv2.imread(img_path)
        if img is None:
            # Fallback if image can't be read
            return Image.open(img_path).convert('RGB')
        
        # Crop black borders
        img = crop_black_borders(img)
        
        # Apply skin masking
        skin_mask = create_skin_mask(img)
        
        # Process based on mask type
        if self.mask_type == 'binary':
            # Create a new image with mask applied
            masked_img = img.copy()
            # Set non-skin areas to background color (default: black)
            bg_color = self.mask_background[::-1]  # Convert RGB to BGR for OpenCV
            masked_img[~skin_mask] = bg_color
            
        elif self.mask_type == 'focus':
            # Extract minimum bounding box of skin region
            rows = np.any(skin_mask, axis=1)
            cols = np.any(skin_mask, axis=0)
            
            if np.sum(rows) == 0 or np.sum(cols) == 0:
                # Fallback if mask is empty
                masked_img = img
            else:
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # Add some padding (10% of dimensions)
                h, w = img.shape[:2]
                pad_h = int(h * 0.1)
                pad_w = int(w * 0.1)
                
                rmin = max(0, rmin - pad_h)
                rmax = min(h - 1, rmax + pad_h)
                cmin = max(0, cmin - pad_w)
                cmax = min(w - 1, cmax + pad_w)
                
                # Crop to skin region
                masked_img = img[rmin:rmax+1, cmin:cmax+1]
                
        elif self.mask_type == 'highlight':
            # Create a highlighted image
            masked_img = img.copy()
            
            # Slightly dim non-skin areas
            masked_img[~skin_mask] = (masked_img[~skin_mask] * 0.5).astype(np.uint8)
            
            # Optional: add colored border around skin areas
            # This can help visualize the mask without losing context
            kernel = np.ones((3, 3), np.uint8)
            mask_border = cv2.dilate(skin_mask.astype(np.uint8), kernel) - skin_mask.astype(np.uint8)
            masked_img[mask_border == 1] = [0, 255, 0]  # Green border around skin
        
        else:
            # Fallback to original image
            masked_img = img
        
        # Convert from BGR to RGB and then to PIL Image
        masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(masked_img_rgb)
        
        return pil_image
    
    def _process_and_save_image(self, img_path, save_path):
        """Process image and save the result for future use."""
        pil_image = self._process_image(img_path)
        
        # Save the processed image
        pil_image.save(save_path)
        
        return pil_image
    
    def preprocess_all_images(self, num_workers=4):
        """
        Preprocess and save all images in the dataset.
        Useful to run before training to speed up data loading.
        
        Args:
            num_workers: Number of parallel workers
        """
        if not self.preprocess_dir:
            raise ValueError("preprocess_dir must be set to use this function")
        
        if not self.apply_mask:
            print("Masking is disabled, no preprocessing needed")
            return
        
        os.makedirs(self.preprocess_dir, exist_ok=True)
        
        # Collect all image paths
        image_paths = []
        save_paths = []
        for idx in range(len(self.df)):
            image_name = self.df.iloc[idx]['image_name']
            if not image_name.endswith('.jpg'):
                image_name = f"{image_name}.jpg"
            
            img_path = os.path.join(self.image_dir, image_name)
            save_path = os.path.join(self.preprocess_dir, image_name)
            
            if not os.path.exists(save_path):
                image_paths.append(img_path)
                save_paths.append(save_path)
        
        if not image_paths:
            print("All images are already preprocessed")
            return
        
        print(f"Preprocessing {len(image_paths)} images with {num_workers} workers...")
        
        # Create processing arguments - include mask_type
        process_args = [(img_path, save_path, self.mask_type) 
                        for img_path, save_path in zip(image_paths, save_paths)]
        
        # If using only 1 worker, avoid multiprocessing entirely
        if num_workers == 1:
            for args in tqdm(process_args):
                process_single_image(args)
        else:
            # Use multiprocessing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(process_single_image, process_args), 
                                total=len(process_args)))
            
            successful = sum(1 for result in results if result)
            print(f"Successfully preprocessed {successful} of {len(process_args)} images")
        
        print("Preprocessing complete!")


def process_single_image(args):
    """
    Process a single image with masking.
    This function needs to be at module level for multiprocessing to work.
    
    Args:
        args: Tuple containing (img_path, save_path, mask_type)
    
    Returns:
        True on success
    """
    img_path, save_path, mask_type = args
    
    # Read the image with OpenCV
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read {img_path}")
        return False
    
    # Crop black borders
    img = crop_black_borders(img)
    
    # Apply skin masking
    skin_mask = create_skin_mask(img)
    
    # Process based on mask type
    if mask_type == 'binary':
        # Create a new image with mask applied
        masked_img = img.copy()
        # Set non-skin areas to background color (default: black)
        masked_img[~skin_mask] = (0, 0, 0)  # Black background
        
    elif mask_type == 'focus':
        # Extract minimum bounding box of skin region
        rows = np.any(skin_mask, axis=1)
        cols = np.any(skin_mask, axis=0)
        
        if np.sum(rows) == 0 or np.sum(cols) == 0:
            # Fallback if mask is empty
            masked_img = img
        else:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add some padding (10% of dimensions)
            h, w = img.shape[:2]
            pad_h = int(h * 0.1)
            pad_w = int(w * 0.1)
            
            rmin = max(0, rmin - pad_h)
            rmax = min(h - 1, rmax + pad_h)
            cmin = max(0, cmin - pad_w)
            cmax = min(w - 1, cmax + pad_w)
            
            # Crop to skin region
            masked_img = img[rmin:rmax+1, cmin:cmax+1]
            
    elif mask_type == 'highlight':
        # Create a highlighted image
        masked_img = img.copy()
        
        # Slightly dim non-skin areas
        masked_img[~skin_mask] = (masked_img[~skin_mask] * 0.5).astype(np.uint8)
        
        # Optional: add colored border around skin areas
        kernel = np.ones((3, 3), np.uint8)
        mask_border = cv2.dilate(skin_mask.astype(np.uint8), kernel) - skin_mask.astype(np.uint8)
        masked_img[mask_border == 1] = [0, 255, 0]  # Green border around skin
    
    else:
        # Fallback to original image
        masked_img = img
    
    # Save the masked image
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        cv2.imwrite(save_path, masked_img)
        return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
        return False

# Function to create masked data loaders
def prepare_masked_data(
    csv_path, 
    image_dir, 
    apply_mask=True, 
    mask_type='binary',
    preprocess_dir=None,
    batch_size=64,
    test_batch_size=64,
    val_size=0.15,
    test_size=0.15,
    seed=42
):
    """
    Prepare data loaders with masked images.
    
    Args:
        csv_path: Path to CSV with image names and labels
        image_dir: Directory containing images
        apply_mask: Whether to apply skin masking
        mask_type: Type of masking ('binary', 'focus', or 'highlight')
        preprocess_dir: Directory to save preprocessed images
        batch_size: Batch size for training
        test_batch_size: Batch size for validation/testing
        val_size: Proportion of data for validation
        test_size: Proportion of data for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, class_counts
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Display dataset information
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    monk_scale_counts = df['monk_scale_type'].value_counts().sort_index()
    for scale, count in monk_scale_counts.items():
        print(f"  Monk Scale {scale}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Split the data into train, validation, and test sets
    from sklearn.model_selection import train_test_split
    
    # First, split off the test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['monk_scale_type'],
        random_state=seed
    )
    
    # Then split the train_val set into train and validation
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size / (1 - test_size),  # Adjusted validation size
        stratify=train_val_df['monk_scale_type'],
        random_state=seed
    )
    
    print(f"\nTraining set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Get transformations
    train_transform, val_transform = get_transforms(config.MODEL_TYPE)
    
    # If using 'focus' mask type, we may need different augmentations
    # since the images will be cropped to skin regions
    if mask_type == 'focus':
        # Consider using different augmentations for focus masks
        # For example, less aggressive cropping since images are already cropped
        pass
    
    # Create datasets with masking
    train_dataset = MaskedMonkScaleDataset(
        train_df, 
        image_dir, 
        transform=train_transform,
        apply_mask=apply_mask,
        mask_type=mask_type,
        preprocess_dir=preprocess_dir
    )
    
    val_dataset = MaskedMonkScaleDataset(
        val_df, 
        image_dir, 
        transform=val_transform,
        apply_mask=apply_mask,
        mask_type=mask_type,
        preprocess_dir=preprocess_dir
    )
    
    test_dataset = MaskedMonkScaleDataset(
        test_df, 
        image_dir, 
        transform=val_transform,
        apply_mask=apply_mask,
        mask_type=mask_type,
        preprocess_dir=preprocess_dir
    )
    
    # Optional: Preprocess all images before training
    if preprocess_dir:
        print("Preprocessing training images...")
        train_dataset.preprocess_all_images()
        print("Preprocessing validation images...")
        val_dataset.preprocess_all_images()
        print("Preprocessing test images...")
        test_dataset.preprocess_all_images()
    
    # Create a weighted sampler for training
    # Calculate weights for each sample
    class_counts = train_df['monk_scale_type'].value_counts().sort_index()
    weights = 1.0 / class_counts[train_df['monk_scale_type'].values].values
    train_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,  # Use weighted sampler for balanced batches
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader, test_loader, monk_scale_counts

def save_metrics_to_file(metrics_dict, model_name, output_dir, test_size=None):
    """
    Save evaluation metrics to a text file.
    
    Args:
        metrics_dict: Dictionary of metrics
        model_name: Name of the model (e.g., 'EfficientNetB0', 'EfficientNetB3')
        output_dir: Directory to save the file
        test_size: Size of the test dataset (optional)
    
    Returns:
        file_path: Path to the saved metrics file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"{model_name}_metrics_{timestamp}.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"=== {model_name} Evaluation Metrics ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if test_size:
            f.write(f"Test Dataset Size: {test_size}\n\n")
        
        # Write general metrics
        f.write("=== General Metrics ===\n")
        for metric_name, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric_name}: {value:.4f}\n")
            elif metric_name == 'confusion_matrix' or metric_name == 'class_predictions':
                # Skip these for now, we'll handle them separately
                continue
            else:
                f.write(f"{metric_name}: {value}\n")
        
        # If confusion matrix is in the dict
        if 'confusion_matrix' in metrics_dict:
            f.write("\n=== Confusion Matrix ===\n")
            cm = metrics_dict['confusion_matrix']
            # Format confusion matrix for text output
            f.write(np.array2string(cm, precision=2))
            f.write("\n")
        
        # If prediction distribution is in the dict
        if 'class_predictions' in metrics_dict:
            f.write("\n=== Prediction Distribution ===\n")
            total_predictions = sum(metrics_dict['class_predictions'].values())
            for class_idx, count in sorted(metrics_dict['class_predictions'].items()):
                percentage = (count / total_predictions) * 100
                f.write(f"Monk Scale {class_idx+1}: {count} predictions ({percentage:.2f}%)\n")
    
    print(f"Metrics saved to {file_path}")
    return file_path

def save_model_architecture(model, model_name, output_dir, include_params=True):
    """
    Save model architecture details to a text file.
    
    Args:
        model: PyTorch model
        model_name: Name of the model (e.g., 'EfficientNetB0', 'EfficientNetB3')
        output_dir: Directory to save the file
        include_params: Whether to include parameter counts and shapes
    
    Returns:
        file_path: Path to the saved architecture file
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"{model_name}_architecture_{timestamp}.txt")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(f"=== {model_name} Architecture ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write model structure
        f.write("=== Model Structure ===\n")
        f.write(str(model))
        f.write("\n\n")
        
        if include_params:
            f.write("=== Layer Parameters ===\n")
            total_params = 0
            trainable_params = 0
            
            # For each named parameter
            for name, param in model.named_parameters():
                param_count = param.numel()
                f.write(f"{name}: Shape {param.shape}, Parameters: {param_count:,}")
                
                # Check if parameter requires gradients (trainable)
                if param.requires_grad:
                    f.write(" (trainable)\n")
                    trainable_params += param_count
                else:
                    f.write(" (frozen)\n")
                
                total_params += param_count
            
            f.write(f"\nTotal Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)\n")
            f.write(f"Frozen Parameters: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)\n")
            
            # Try to add feature extractor and classifier details if they exist
            try:
                if hasattr(model, 'feature_extractor'):
                    f.write("\n=== Feature Extractor ===\n")
                    f.write(str(model.feature_extractor))
                
                if hasattr(model, 'classifier'):
                    f.write("\n\n=== Classifier ===\n")
                    f.write(str(model.classifier))
            except:
                # Silently continue if these components don't exist
                pass
    
    print(f"Model architecture saved to {file_path}")
    return file_path

# Example transformations for masked images
def get_transforms(model_type='b0'):
    """
    Create transformations for masked images.
    Potentially different based on mask type.
    
    Args:
        model_type: 'b0', 'b3', or 'b5'
    """
    # Set image size based on model type
    if model_type == 'b0':
        image_size = 224
    elif model_type == 'b3':
        image_size = 300
    elif model_type == 'b5':
        image_size = 456
    else:
        raise ValueError("model_type must be one of 'b0', 'b3', or 'b5'")
    
    # Training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(config.RANDOMROTATION),
        transforms.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST, saturation=config.SATURATION, hue=config.HUE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    # Validation/Test transformations (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    return train_transform, val_transform

# Function to create a weighted sampler for imbalanced classes
def create_weighted_sampler(df):
    # Get class counts
    class_counts = df['monk_scale_type'].value_counts().sort_index()
    
    # Calculate weights for each sample
    weights = 1.0 / class_counts[df['monk_scale_type'].values].values
    
    # Create sampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    return sampler

# Training function with device parameter
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, device=None, output_dir=None):
    # Default values if not provided
    if device is None:
        device = config.DEVICE
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    since = time.time()
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(
        patience = config.PATIENCE,
        verbose = True
    )

    # Track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    
    # Keep track of progress
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_balanced_acc': [], 'val_balanced_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Iterate over batches
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)  # Using device parameter
                labels = labels.to(device)  # Using device parameter
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
            epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0, labels=range(7)
            )
            
            # Record all metrics
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            history[f'{phase}_balanced_acc'].append(epoch_balanced_acc)
            history[f'{phase}_precision'].append(precision)
            history[f'{phase}_recall'].append(recall)
            history[f'{phase}_f1'].append(f1)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
                  f'Balanced Acc: {epoch_balanced_acc:.4f} Precision: {precision:.4f} '
                  f'Recall: {recall:.4f} F1: {f1:.4f}')
            
            # Deep copy the model if it's the best validation performance so far
            if phase == 'val':
                # Update the learning rate scheduler based on validation metrics
                if scheduler is not None:
                    scheduler.step(epoch_loss)

                # Early stopping and model checkpointing
                early_stopping(epoch_loss)
                    
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # Save the best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'balanced_acc': epoch_balanced_acc,
                    }, os.path.join(output_dir, 'best_model.pth'))

        # Check for early stopping
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

        print()
    
    # Save the final model
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'balanced_acc': epoch_balanced_acc,
    }, os.path.join(output_dir, 'final_model.pth'))
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val balanced loss: {best_loss:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training progress
    plot_training_history(history, output_dir=output_dir)
    
    return model, history


# Evaluate model with device parameter
def evaluate_model(model, test_loader, device=None, output_dir=None):
    # Default values if not provided
    if device is None:
        device = config.DEVICE
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    model.eval()
    all_preds = []
    all_labels = []

    # Add counters for all classes (0-9 representing Monk Scale 1-10)
    class_predictions = {i: 0 for i in range(10)}

    # For additional analysis
    unseen_confidences = []
    unseen_origins = {i: 0 for i in range(7)}
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)  # Using device parameter
            labels = labels.to(device)  # Using device parameter
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Confidence analysis
            confidences = torch.softmax(outputs, dim=1)
            
            # Process each prediction
            for i, (pred, true_label) in enumerate(zip(preds.cpu().numpy(), labels.cpu().numpy())):
                # Count all class predictions
                class_predictions[pred] += 1
                
                # Confidence analysis for unseen classes
                if pred >= 7:  # If predicted as unseen class
                    conf = confidences[i, pred].item()
                    unseen_confidences.append(conf)
                    
                    # Confusion analysis - which true classes get confused as unseen
                    if true_label < 7:  # If the true class is known
                        unseen_origins[true_label] += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print prediction distribution
    print("\nPrediction Distribution:")
    total_predictions = len(all_preds)
    for class_idx in range(10):
        count = class_predictions[class_idx]
        percentage = (count / total_predictions) * 100
        if class_idx < 7:
            print(f"Monk Scale {class_idx+1}: {count} predictions ({percentage:.2f}%) - In training data")
        else:
            print(f"Monk Scale {class_idx+1}: {count} predictions ({percentage:.2f}%) - UNSEEN CLASS")
    
    # Print unseen class predictions summary
    unseen_count = sum(class_predictions[i] for i in range(7, 10))
    unseen_percentage = (unseen_count / total_predictions) * 100
    print(f"\nTotal Unseen Classes (8-10): {unseen_count} predictions ({unseen_percentage:.2f}%)")
    
    # Confidence analysis
    if unseen_confidences:
        avg_confidence = sum(unseen_confidences) / len(unseen_confidences)
        print(f"\nConfidence Analysis for Unseen Classes (8-10):")
        print(f"  Average confidence: {avg_confidence:.4f}")
        print(f"  Min confidence: {min(unseen_confidences):.4f}")
        print(f"  Max confidence: {max(unseen_confidences):.4f}")
    else:
        print("\nNo predictions for unseen classes (8-10)")
    
    # Confusion analysis
    print("\nTrue classes predicted as unseen classes (8-10):")
    any_confused = False
    for class_idx, count in unseen_origins.items():
        if count > 0:
            any_confused = True
            print(f"  Monk Scale {class_idx+1}: {count} instances")
    
    if not any_confused:
        print("  None")

    # Calculate metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Calculate macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = \
        precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0, labels=range(7))
    
    # Calculate weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = \
        precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0, labels=range(7))
    
    print(f'\nTest Accuracy: {acc:.4f}')
    print(f'Test Balanced Accuracy: {balanced_acc:.4f}')
    print(f'Test Macro Precision: {precision_macro:.4f}')
    print(f'Test Macro Recall: {recall_macro:.4f}')
    print(f'Test Macro F1: {f1_macro:.4f}')
    print(f'Test Weighted Precision: {precision_weighted:.4f}')
    print(f'Test Weighted Recall: {recall_weighted:.4f}')
    print(f'Test Weighted F1: {f1_weighted:.4f}')
    
    # Classification report
    class_names = [f'Scale {i+1}' for i in range(10)]
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, 
                               target_names=class_names[:7],  # Only first 7 classes
                               labels=range(7)))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(7))
    plot_confusion_matrix(cm, class_names[:7], output_dir=output_dir)

    if output_dir:
        metrics_dict = {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'class_predictions': class_predictions,
            'unseen_confidences': unseen_confidences if unseen_confidences else None,
            'unseen_origins': unseen_origins,
        }
        
        metrics_file = save_metrics_to_file(
            metrics_dict,
            model_name="TestResults", 
            output_dir=output_dir,
            test_size=len(test_loader.dataset)
        )
        print(f"Metrics saved to: {metrics_file}")

    print("\nNote: This model is designed for 10 classes (Monk Scale 1-10)")
    print("but trained and evaluated on 7 classes (Monk Scale 1-7) due to data availability.")
    print("Predictions for classes 8-10 should be considered experimental.")
    
    return acc, balanced_acc, precision_macro, recall_macro, f1_macro, cm


# Update plot confusion matrix to accept output_dir
def plot_confusion_matrix(cm, class_names, output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


# Update plot training history to accept output_dir
def plot_training_history(history, output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_balanced_acc'], label='Training Balanced Accuracy')
    plt.plot(history['val_balanced_acc'], label='Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title('Training and Validation Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot precision and recall
    plt.subplot(2, 2, 3)
    plt.plot(history['train_precision'], label='Training Precision')
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.plot(history['train_recall'], label='Training Recall')
    plt.plot(history['val_recall'], label='Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision and Recall')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 score
    plt.subplot(2, 2, 4)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()


# Plot class distribution
def plot_class_distribution(class_counts):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.xlabel('Monk Scale Type')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in the Dataset')
    plt.xticks(range(len(class_counts.index)), class_counts.index)
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'class_distribution.png'))
    #plt.show()
    plt.close()


# Main function
def main(config):
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Using mask type: {config.MASK_TYPE}")
    print(f"Using model type: {config.MODEL_TYPE}")
    
    # Prepare masked data
    train_loader, val_loader, test_loader, monk_scale_counts = prepare_masked_data(
        csv_path=config.CSV_PATH,
        image_dir=config.IMAGE_DIR,
        apply_mask=config.APPLY_MASK,
        mask_type=config.MASK_TYPE,
        preprocess_dir=config.PREPROCESS_DIR,
        batch_size=config.BATCH_SIZE,
        test_batch_size=config.TEST_BATCH_SIZE,
        val_size=config.VAL_SIZE,
        test_size=config.TEST_SIZE
    )
    
    # Create model based on model type
    if config.MODEL_TYPE == 'b0':
        model = EfficientNetB0SkinToneClassifier(
            num_classes=config.NUM_CLASSES,
            freeze_features=config.FREEZE_FEATURES
        )
    elif config.MODEL_TYPE == 'b3':
        model = EfficientNetB3SkinToneClassifier(
            num_classes=config.NUM_CLASSES,
            freeze_features=config.FREEZE_FEATURES
        )
    elif config.MODEL_TYPE == 'b5':
        model = EfficientNetB5SkinToneClassifier(
            num_classes=config.NUM_CLASSES,
            freeze_features=config.FREEZE_FEATURES
        )
    
    model = model.to(config.DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    architecture_file = save_model_architecture(
        model, 
        model_name=f"EfficientNet{config.MODEL_TYPE}_{config.MASK_TYPE}", 
        output_dir=config.OUTPUT_DIR, 
        include_params=True
    )
    
    # Train model using your existing train_model function
    dataloaders = {'train': train_loader, 'val': val_loader}
    model, history = train_model(
        model, 
        dataloaders, 
        criterion, 
        optimizer, 
        scheduler,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR
    )
    
    # Evaluate on test set using your existing evaluate_model function
    acc, balanced_acc, precision_macro, recall_macro, f1_macro, cm = evaluate_model(model, 
                                                                                    test_loader, 
                                                                                    device=config.DEVICE,
                                                                                    output_dir=config.OUTPUT_DIR)

    # After evaluation, save comprehensive experiment summary
    experiment_metrics = {
        'model_type': config.MODEL_TYPE,
        'mask_type': config.MASK_TYPE if config.APPLY_MASK else "original",
        'freeze_features': config.FREEZE_FEATURES,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.NUM_EPOCHS,
        'test_accuracy': acc,
        'test_balanced_accuracy': balanced_acc,
        'test_f1_macro': f1_macro,
        # You can also add these:
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }
    
    summary_file = save_metrics_to_file(
        experiment_metrics,
        model_name=f"ExperimentSummary_{config.MODEL_TYPE}_{config.MASK_TYPE if config.APPLY_MASK else 'original'}", 
        output_dir=config.OUTPUT_DIR,
        test_size=len(test_loader.dataset)
    )
    print(f"Experiment summary saved to: {summary_file}")
    
    print("Training and evaluation complete!")
    print(f"Best model saved to {os.path.join(config.OUTPUT_DIR, 'best_model.pth')}")
    
    return model, history


if __name__ == "__main__":
    config = ConfigB0()
    main(config)