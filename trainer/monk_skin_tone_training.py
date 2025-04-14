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
from torch_optimizer import RAdam, Lookahead
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, 
                             classification_report, 
                             balanced_accuracy_score, 
                             precision_recall_fscore_support, 
                             mean_absolute_error,
                             mean_squared_error, 
                             cohen_kappa_score, 
                             roc_auc_score,
                             roc_curve, 
                             auc)
from sklearn.preprocessing import label_binarize
from scipy.stats import spearmanr
from PIL import Image
import copy
import time
import collections
import cv2

# Import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import (EfficientNetB0SkinToneClassifier, 
                                    EfficientNetB3SkinToneClassifier,
                                    EfficientNetB5SkinToneClassifier, 
                                    AttentionSkinToneClassifier, 
                                    SkinToneFusionModel)

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
    CSV_PATH = os.path.join(DATA_DIR, "deduplicated_monk_scale_dataset.csv")
    
    # Common settings
    NUM_CLASSES = 7
    FREEZE_FEATURES = True
    
    # Data split
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Device settings
    GPU_ID = 2
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    # Common training parameters
    BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-6  # Reduced for fine-tuning
    WEIGHT_DECAY = 1e-5 
    PATIENCE = 10

    # Transformation parameters
    RANDOMROTATION = 10
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    USE_LAB = True

    TEST_NUM = 18


# EfficientNet B0 Config (224x224)
class ConfigB0(BaseConfig):
    MODEL_TYPE = 'b0'
    IMAGE_DIMENSION = 224
    
    # Specific directories for B0
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_processed")
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_test{BaseConfig.TEST_NUM}")


# EfficientNet B3 Config (300x300)
class ConfigB3(BaseConfig):
    MODEL_TYPE = 'b3'
    IMAGE_DIMENSION = 300
    
    # Specific directories for B3
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_processed")
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_test{BaseConfig.TEST_NUM}")
    
    # Slightly different training params (optional)
    BATCH_SIZE = 48  # Reduced batch size for larger model


# EfficientNet B5 Config (456x456)
class ConfigB5(BaseConfig):
    MODEL_TYPE = 'b5'
    IMAGE_DIMENSION = 456
    
    # Specific directories for B5
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_processed")
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_test{BaseConfig.TEST_NUM}")
    
    # Adjusted parameters for the larger model
    BATCH_SIZE = 32  # Smaller batch size
    LEARNING_RATE = 0.00005  # Even smaller learning rate

# EfficientNet B5 Config (456x456)
class VGG_16(BaseConfig):
    MODEL_TYPE = 'VGG'
    IMAGE_DIMENSION = 224
    
    # Specific directories for B5
    IMAGE_DIR = os.path.join(BaseConfig.DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_processed")
    OUTPUT_DIR = os.path.join(BaseConfig.PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"VGG_16_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_test{BaseConfig.TEST_NUM}")
    
    # Adjusted parameters for the larger model
    BATCH_SIZE = 48  # Smaller batch size
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

def to_lab(image):
    """
    Convert an RGB image to CIELAB color space.
    
    Args:
        image: RGB image tensor with values in [0, 1] and shape (C, H, W)
        
    Returns:
        LAB image tensor with values in [0, 1] and shape (C, H, W)
    """
    # Convert from tensor format to numpy for OpenCV
    if isinstance(image, torch.Tensor):
        # Move channels to last dimension and convert to numpy
        image_np = image.permute(1, 2, 0).cpu().numpy()
        # Scale to [0, 255] for OpenCV
        image_np = (image_np * 255.0).astype(np.uint8)
    else:
        # If it's already a numpy array, ensure it's in the right range
        image_np = (image * 255.0).astype(np.uint8)
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    
    # Normalize LAB values to [0, 1] range
    lab_image = lab_image.astype(np.float32)
    # L channel is in [0, 255], a and b channels are in [0, 255] but centered at 128
    lab_image[:,:,0] = lab_image[:,:,0] / 255.0  # L channel
    lab_image[:,:,1] = (lab_image[:,:,1] + 128) / 255.0  # a channel (now [0, 1])
    lab_image[:,:,2] = (lab_image[:,:,2] + 128) / 255.0  # b channel (now [0, 1])
    
    # Convert back to tensor with channels first
    lab_tensor = torch.from_numpy(lab_image).permute(2, 0, 1).float()
    return lab_tensor

# Create the dataset class
class MonkScaleDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, cache_size=3000):
        """
        Dataset for Monk Scale images.
        
        Args:
            df (pandas.DataFrame): DataFrame with image_name and monk_scale_type
            image_dir (string): Directory with all the images
            transform (callable, optional): Transform to be applied on the images
            cache_size (int): Maximum size of the LRU cache
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.cache_size = cache_size
        
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
        
        # Check if image is in cache
        if img_path in self.cache:
            # Get image from cache and mark as recently used by moving to end
            image = self.cache.pop(img_path)
            self.cache[img_path] = image
        else:
            # Load image
            image = Image.open(img_path).convert('RGB')
                
            # Add to cache if caching is enabled
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)  # Remove least recently used
            self.cache[img_path] = image
        
        # Apply transformations
        if self.transform:
            # Ensure the transform works with PIL Image
            transformed_image = self.transform(image)
        else:
            transformed_image = image
        
        # Get label
        label = self.df.iloc[idx]['monk_scale_type'] - 1
        
        return transformed_image, label

def prepare_data(
    csv_path, 
    image_dir, 
    batch_size=64,
    test_batch_size=64,
    val_size=0.15,
    test_size=0.15,
    seed=42,
    augmentation_method="weighted_sampling"  # Options: "weighted_sampling", "duplicate_samples", "both"
):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Display dataset information
    print(f"Dataset shape: {df.shape}")
    print("\nOriginal Class distribution:")
    monk_scale_counts = df['monk_scale_type'].value_counts().sort_index()
    for scale, count in monk_scale_counts.items():
        print(f"  Monk Scale {scale}: {count} samples ({count/len(df)*100:.2f}%)")
    
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

    # Decide on augmentation approach
    if augmentation_method == "duplicate_samples" or augmentation_method == "both":
        print("\nApplying physical augmentation by duplicating samples...")
        train_df = augment_rare_classes(train_df, min_samples=1500, multiplier=6)
        
        # Update class counts after augmentation
        monk_scale_counts = train_df['monk_scale_type'].value_counts().sort_index()
        
        print("\nClass distribution after physical augmentation:")
        for scale, count in monk_scale_counts.items():
            print(f"  Monk Scale {scale}: {count} samples ({count/len(train_df)*100:.2f}%)")

    # Display dataset information
    print(f"\nTraining set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Get transformations
    train_transform, val_transform = get_transforms(config.MODEL_TYPE, use_lab=config.USE_LAB)

    save_transformation_details(train_transform, output_dir=config.OUTPUT_DIR)
    
    # Create datasets
    train_dataset = MonkScaleDataset(
        train_df, 
        image_dir, 
        transform=train_transform
    )
    
    val_dataset = MonkScaleDataset(
        val_df, 
        image_dir, 
        transform=val_transform
    )
    
    test_dataset = MonkScaleDataset(
        test_df, 
        image_dir, 
        transform=val_transform
    )
    
    # Initialize class weights for potential use
    class_weights = None
    
    # Decide whether to use weighted sampling
    if augmentation_method == "weighted_sampling" or augmentation_method == "both":
        print("\nApplying weighted sampling strategy...")
        
        # Calculate inverse class weights with boosting for extreme classes
        class_weights = {}
        total_samples = len(train_df)
        
        # Adjust weights based on whether we're using both methods
        boost_factor_1 = 2.5 if augmentation_method == "weighted_sampling" else 1.5  # Reduced if using both
        boost_factor_7 = 3.0 if augmentation_method == "weighted_sampling" else 2.0  # Reduced if using both
        boost_factor_mid = 1.5 if augmentation_method == "weighted_sampling" else 1.2  # Reduced if using both
        
        # Calculate weights
        for cls, count in monk_scale_counts.items():
            # Base weight is inverse of frequency
            weight = total_samples / (count * len(monk_scale_counts))
            
            # Boost extreme classes
            if cls == 1:  # Class 1
                weight *= boost_factor_1
            elif cls == 7:  # Class 7
                weight *= boost_factor_7
            elif cls in [5, 6]:  # Classes 5-6
                weight *= boost_factor_mid
                
            class_weights[cls] = weight
        
        # Normalize weights
        weight_sum = sum(class_weights.values())
        for cls in class_weights:
            class_weights[cls] = class_weights[cls] * len(class_weights) / weight_sum
        
        print("\nClass Weights for Sampling:")
        for cls, weight in sorted(class_weights.items()):
            print(f"  Class {cls}: {weight:.4f}")
        
        # Apply weights to each sample
        sample_weights = [class_weights[cls] for cls in train_df['monk_scale_type']]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_df),
            replacement=True
        )
        
        # Create dataloader with weighted sampler
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    else:
        # Use regular shuffling
        print("\nUsing regular shuffling (no weighted sampling)...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,  
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    # Create validation and test loaders (no need for weighted sampling)
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
    
    return train_loader, val_loader, test_loader, monk_scale_counts, class_weights

def save_metrics_to_file(metrics_dict, model_name, output_dir, test_size=None):
    """
    Save evaluation metrics to a text file.
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

        f.write("\n=== Ordinal Classification Metrics ===\n")
        if 'distance_weighted_accuracy' in metrics_dict:
            f.write(f"Distance-weighted Accuracy: {metrics_dict['distance_weighted_accuracy']:.4f}\n")
        if 'off_by_one_accuracy' in metrics_dict:
            f.write(f"Off-by-one Accuracy: {metrics_dict['off_by_one_accuracy']:.4f}\n")
        if 'spearman_correlation' in metrics_dict:
            f.write(f"Spearman Rank Correlation: {metrics_dict['spearman_correlation']:.4f}\n")
        
        # Add similar classes confusion rates
        if 'similar_classes_confusion' in metrics_dict:
            f.write("\n=== Similar Classes Confusion Rates ===\n")
            for pair, rate in metrics_dict['similar_classes_confusion'].items():
                f.write(f"Monk Scale {pair}: {rate:.4f}\n")
        
        f.write("\n=== Calibration Metrics ===\n")
        if 'expected_calibration_error' in metrics_dict:
            f.write(f"Expected Calibration Error: {metrics_dict['expected_calibration_error']:.4f}\n")
        if 'brier_score' in metrics_dict:
            f.write(f"Brier Score: {metrics_dict['brier_score']:.4f}\n")
    
    print(f"Metrics saved to {file_path}")
    return file_path

def save_model_architecture(model, model_name, output_dir, include_params=True):
    """
    Save model architecture details to a text file.
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

def analyze_data_distribution(train_loader, val_loader, test_loader, num_classes=10, output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
   
    # Initialize counters for each dataloader
    train_counts = {i: 0 for i in range(num_classes)}
    val_counts = {i: 0 for i in range(num_classes)}
    test_counts = {i: 0 for i in range(num_classes)}
   
    # Collect counts for training data
    for _, labels in tqdm(train_loader, desc='Analyzing training data'):
        for label in labels.numpy():
            train_counts[label] += 1
   
    # Collect counts for validation data
    for _, labels in tqdm(val_loader, desc='Analyzing validation data'):
        for label in labels.numpy():
            val_counts[label] += 1
   
    # Collect counts for test data
    for _, labels in tqdm(test_loader, desc='Analyzing test data'):
        for label in labels.numpy():
            test_counts[label] += 1
   
    # Calculate totals
    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())
    test_total = sum(test_counts.values())
   
    # Print results
    print("=== Class Distribution Analysis ===")
    print(f"{'Class':<10} {'Training':<20} {'Validation':<20} {'Testing':<20}")
    print("-" * 70)
   
    for class_idx in range(num_classes):
        # Calculate percentages
        train_pct = (train_counts[class_idx] / train_total * 100) if train_total > 0 else 0
        val_pct = (val_counts[class_idx] / val_total * 100) if val_total > 0 else 0
        test_pct = (test_counts[class_idx] / test_total * 100) if test_total > 0 else 0
       
        # Format for display
        train_str = f"{train_counts[class_idx]} ({train_pct:.2f}%)"
        val_str = f"{val_counts[class_idx]} ({val_pct:.2f}%)"
        test_str = f"{test_counts[class_idx]} ({test_pct:.2f}%)"
       
        # Print to console
        print(f"Monk {class_idx+1:<5} {train_str:<20} {val_str:<20} {test_str:<20}")
   
    print("-" * 70)
    print(f"Total     {train_total:<20} {val_total:<20} {test_total:<20}")

def calculate_class_weights(train_df):
    """
    Calculate class weights with a power to smooth out extreme differences
    
    Args:
        train_df: Training DataFrame with monk_scale_type column
    
    Returns:
        Normalized class weights
    """
    class_counts = train_df['monk_scale_type'].value_counts()
    total_samples = len(train_df)
    
    # Use inverse frequency with a power to smooth out extreme differences
    weights = (total_samples / (len(class_counts) * class_counts)) ** 0.5
    
    # Normalize weights to sum to len(class_counts)
    weights = weights / weights.sum() * len(weights)
    
    return weights
    

def augment_rare_classes(train_df, min_samples=1500, multiplier=6, preserve_classes=None):
    """
    Augment rare classes using a multiplier approach
    
    Args:
        train_df: Training DataFrame
        min_samples: Minimum number of samples to trigger augmentation
        multiplier: How many times to multiply samples for rare classes
        preserve_classes: List of classes to preserve original samples (e.g., [1])
    
    Returns:
        Augmented DataFrame
    """
    # Create a copy of the dataframe to avoid modifying the original
    augmented_df = train_df.copy()
    
    # Get current class distribution
    class_counts = train_df['monk_scale_type'].value_counts().sort_index()
    
    # Default to preserving class 1 if not specified
    if preserve_classes is None:
        preserve_classes = [1]
    
    print("\nDetailed Augmentation Process:")
    for scale, count in class_counts.items():
        # Skip classes that should be preserved
        if scale in preserve_classes:
            print(f"Monk Scale {scale}: Preserved (original {count} samples)")
            continue
        
        # Check if class needs augmentation
        if count < min_samples:
            # Find rows for this scale
            scale_rows = train_df[train_df['monk_scale_type'] == scale]
            
            # Calculate number of samples to add
            target_total = min(count * multiplier, min_samples)
            samples_to_add = target_total - count
            
            print(f"Monk Scale {scale}:")
            print(f"  Original samples: {count}")
            print(f"  Target total samples: {target_total}")
            print(f"  Samples to add: {samples_to_add}")
            
            # Sample with replacement
            additional_samples = scale_rows.sample(n=samples_to_add, replace=True)
            
            # Append to augmented dataframe
            augmented_df = pd.concat([augmented_df, additional_samples], ignore_index=True)
        else:
            print(f"Monk Scale {scale}: No augmentation needed (already {count} samples)")
    
    # Verify final distribution
    final_counts = augmented_df['monk_scale_type'].value_counts().sort_index()
    print("\nFinal Distribution:")
    for scale, count in final_counts.items():
        print(f"Monk Scale {scale}: {count} samples")
    
    return augmented_df

# Example transformations for images
def get_transforms(model_type='b0', use_lab=False):
    """
    Create transformations for images.
    
    Args:
        model_type: 'b0', 'b3', or 'b5'
        use_lab: Whether to convert to LAB color space
    """
    
    # Training transformations with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(5),
        
        # Tune color jitter for skin tones
        transforms.ColorJitter(
            brightness=0.05,  # Minimal brightness adjustment
            contrast=0.05,    # Minimal contrast adjustment
            saturation=0.0,   # No saturation change
            hue=0.0           # No hue change (critical for skin tone)
        ),
        
        # Add random resized crops to focus on different areas
        transforms.RandomResizedCrop(
            size=config.IMAGE_DIMENSION,
            scale=(0.85, 1.0),  # Less aggressive crop 
            ratio=(0.9, 1.1)    # Keep aspect ratio close to original
        ),
        
        # Your other augmentations with reduced intensity
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=7
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.ToTensor(),
    ])
    
    # Validation/Test transformations (no augmentation)
    val_transform = transforms.Compose([
        #transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return train_transform, val_transform

def save_transformation_details(transforms_list, output_dir=None):
    """
    Save details about the image transformations used during training.
    
    Args:
        transforms_list: Composition of transforms
        output_dir: Directory to save the file (defaults to config.OUTPUT_DIR)
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"transformation_details_{timestamp}.txt")
    
    with open(file_path, 'w') as f:
        f.write("=== Image Transformation Details ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== Training Transformations ===\n")
        for idx, transform in enumerate(transforms_list.transforms, 1):
            f.write(f"{idx}. {transform.__class__.__name__}\n")
            
            # Add specific details for certain transforms
            if hasattr(transform, 'p'):
                f.write(f"   - Probability: {transform.p}\n")
            
            if isinstance(transform, transforms.RandomRotation):
                f.write(f"   - Degrees: {transform.degrees}\n")
            
            if isinstance(transform, transforms.ColorJitter):
                f.write(f"   - Brightness: {transform.brightness}\n")
                f.write(f"   - Contrast: {transform.contrast}\n")
                f.write(f"   - Saturation: {transform.saturation}\n")
                f.write(f"   - Hue: {transform.hue}\n")
            
            if isinstance(transform, transforms.Resize):
                f.write(f"   - Size: {transform.size}\n")
            
            if isinstance(transform, transforms.RandomPerspective):
                f.write(f"   - Distortion Scale: {transform.distortion_scale}\n")

            if isinstance(transform, transforms.RandomErasing):
                f.write(f"   - Erasing Scale: {transform.scale}\n")
                f.write(f"   - p: {transform.p}\n")

            if isinstance(transform, transforms.RandomAffine):
                f.write(f"   - Degrees: {transform.degrees}\n")
                f.write(f"   - Translate: {transform.translate}\n")
                f.write(f"   - Scale: {transform.scale}\n")
                f.write(f"   - Shear: {transform.shear}\n")
        
        # Additional information
        f.write("\n=== Additional Details ===\n")
        f.write(f"Use LAB Color Space: {config.USE_LAB}\n")
        f.write(f"Image Dimension: {config.IMAGE_DIMENSION}\n")
        f.write(f"Model Type: {config.MODEL_TYPE}\n")
    
    print(f"Transformation details saved to {file_path}")
    return file_path


# Training function
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
    best_balanced_acc = 0.0
    
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
                inputs = inputs.to(device)
                labels = labels.to(device)
                
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
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
                all_labels, all_preds, average='macro', zero_division=0
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
                #early_stopping(-epoch_balanced_acc)
                    
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

                #if epoch_balanced_acc > best_balanced_acc:
                #    best_balanced_acc = epoch_balanced_acc
                #    best_model_wts = copy.deepcopy(model.state_dict())

                    # Save the best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                        'balanced_acc': epoch_balanced_acc,
                        'f1': f1,
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

def plot_training_history(history, output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    # Create a figure with 3 rows of plots
    plt.figure(figsize=(15, 15))
    
    # Plot loss
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(3, 2, 2)
    plt.plot(history['train_balanced_acc'], label='Training Balanced Accuracy')
    plt.plot(history['val_balanced_acc'], label='Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.title('Training and Validation Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot precision and recall
    plt.subplot(3, 2, 3)
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
    plt.subplot(3, 2, 4)
    plt.plot(history['train_f1'], label='Training F1')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot standard accuracy
    plt.subplot(3, 2, 6)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_training_history.png'))
    plt.close()

def mixup_data(x, y, alpha=0.2, device=None):
    """Applies mixup augmentation to a batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    if device is not None:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates loss with mixup targets"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Evaluate model function
def evaluate_model(model, test_loader, device=None, output_dir=None):
    # Default values if not provided
    if device is None:
        device = config.DEVICE
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    # Add counters for all classes (0-9 representing Monk Scale 1-10)
    class_predictions = {i: 0 for i in range(10)}

    # For additional analysis
    unseen_confidences = []
    unseen_origins = {i: 0 for i in range(7)}
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Get probability scores for ROC curve
            probs = torch.softmax(outputs, dim=1)
            all_scores.append(probs.cpu().numpy())

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

    # Concatenate all scores
    all_scores = np.vstack(all_scores)
    
    # Convert labels and predictions to numpy arrays
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

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
        precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    # Calculate weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = \
        precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # This is useful for ordinal classification like skin types
    mae = mean_absolute_error(all_labels_np, all_preds_np)

    mse = mean_squared_error(all_labels_np, all_preds_np)
    
    # Quadratic weighting penalizes predictions that are further from the true label more heavily
    kappa = cohen_kappa_score(all_labels_np, all_preds_np, weights='quadratic')

    # Distance-weighted accuracy (for ordinal relationship)
    dw_acc = distance_weighted_accuracy(all_labels_np, all_preds_np, max_distance=9)
    
    # Off-by-one accuracy
    off_by_one_acc = off_by_one_accuracy(all_labels_np, all_preds_np)
    
    # Confusion rates between similar classes (1-4)
    similar_classes = [0, 1, 2, 3]  # 0-indexed (monk scale 1-4)
    confusion_rates = calculate_similar_classes_confusion(all_labels_np, all_preds_np, similar_classes)
    
    # Spearman's rank correlation
    spearman_corr, p_value = spearmanr(all_labels_np, all_preds_np)

    # Calculate ECE and Brier Score
    ece = compute_ece(all_scores, all_labels_np, n_bins=10)
    brier_score = compute_brier_score(all_scores, all_labels_np, n_classes=config.NUM_CLASSES)

    # Only consider classes 0-6 (monk scales 1-7) that are in the training data
    valid_classes = range(7)
    
    # One-hot encode the labels for AUC calculation
    y_true_bin = label_binarize(all_labels_np, classes=valid_classes)
    
    # Get scores for only the valid classes
    valid_scores = all_scores[:, valid_classes]
    
    # Calculate micro and macro AUC
    try:
        # Micro-average: calculate metrics globally by considering each element
        roc_auc_micro = roc_auc_score(y_true_bin, valid_scores, multi_class='ovr', average='micro')
        
        # Macro-average: calculate metrics for each label, and find their unweighted mean
        roc_auc_macro = roc_auc_score(y_true_bin, valid_scores, multi_class='ovr', average='macro')
        
        # Calculate and plot ROC curve for each class
        plt.figure(figsize=(12, 8))
        
        # Colors for different classes
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, color in zip(range(len(valid_classes)), colors):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], valid_scores[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'ROC curve class {valid_classes[i]+1} (area = {class_auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        roc_auc_micro = None
        roc_auc_macro = None
    
     # Print results including new metrics
    print(f'\nTest Accuracy: {acc:.4f}')
    print(f'Test Balanced Accuracy: {balanced_acc:.4f}')
    print(f'Test Macro Precision: {precision_macro:.4f}')
    print(f'Test Macro Recall: {recall_macro:.4f}')
    print(f'Test Macro F1: {f1_macro:.4f}')
    print(f'Test Weighted Precision: {precision_weighted:.4f}')
    print(f'Test Weighted Recall: {recall_weighted:.4f}')
    print(f'Test Weighted F1: {f1_weighted:.4f}')
    
    # Print new metrics
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Cohen\'s Kappa (Quadratic Weighted): {kappa:.4f}')
    if roc_auc_micro is not None:
        print(f'ROC AUC (micro-average): {roc_auc_micro:.4f}')
    if roc_auc_macro is not None:
        print(f'ROC AUC (macro-average): {roc_auc_macro:.4f}')

    print(f'Distance-weighted Accuracy: {dw_acc:.4f}')
    print(f'Off-by-one Accuracy: {off_by_one_acc:.4f}')
    print(f'Spearman Rank Correlation: {spearman_corr:.4f} (p-value: {p_value:.4f})')
    
    print('\nConfusion Rates between Similar Classes (1-4):')
    for pair, rate in confusion_rates.items():
        print(f'  Monk Scale {pair}: {rate:.4f}')

    print(f'Expected Calibration Error: {ece:.4f}')
    print(f'Brier Score: {brier_score:.4f}')
    
    # Classification report
    class_names = [f'Scale {i+1}' for i in range(10)]
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds, 
                               target_names=class_names[:7],  # Only first 7 classes
                               labels=range(7)))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(7))
    plot_confusion_matrix(cm, class_names[:7], output_dir=output_dir)
    plot_calibration_curve(all_scores, all_labels_np, output_dir=output_dir)

    if output_dir:
        metrics_dict = {
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'mae': mae,
            'mse': mse,
            'cohens_kappa': kappa,
            'roc_auc_micro': roc_auc_micro,
            'roc_auc_macro': roc_auc_macro,
            'distance_weighted_accuracy': dw_acc,
            'off_by_one_accuracy': off_by_one_acc,
            'spearman_correlation': spearman_corr,
            'confusion_matrix': cm,
            'class_predictions': class_predictions,
            'similar_classes_confusion': confusion_rates,
            'unseen_confidences': unseen_confidences if unseen_confidences else None,
            'unseen_origins': unseen_origins,
            'expected_calibration_error': ece,
            'brier_score': brier_score,
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
    
    return acc, balanced_acc, precision_macro, recall_macro, f1_macro, mae, mse, kappa, roc_auc_macro, dw_acc, off_by_one_acc, spearman_corr, ece, brier_score, cm


# Plot confusion matrix function
def plot_confusion_matrix(cm, class_names, output_dir=None):
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
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
    plt.close()

# Define FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
       
    def forward(self, inputs, targets):
        # Use cross_entropy with label smoothing
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none', 
            label_smoothing=self.label_smoothing
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
       
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss

def get_class_weights(monk_scale_count, num_classes=10):
    """
    Calculate class weights, padding with zeros for unseen classes if needed.
    
    Args:
        monk_scale_counts: Counts of existing classes
        num_classes: Total number of classes (including unseen)
    
    Returns:
        Normalized class weights tensor
    """
    # Extend the counts to full number of classes
    full_counts = pd.Series(
        [0] * num_classes, 
        index=range(1, num_classes + 1)
    )
    
    # Update with actual counts
    full_counts.update(monk_scale_count)
    
    # Calculate weights
    total_samples = full_counts.sum()
    class_weights = total_samples / (full_counts * len(full_counts))
    
    # Normalize weights to sum to len(class_weights)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    return torch.FloatTensor(class_weights.values)

class WassersteinLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(WassersteinLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply softmax to get predicted probabilities
        pred_probs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        
        # Compute cumulative distributions
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(targets_one_hot, dim=1)
        
        # Compute Wasserstein distance
        wasserstein_dist = torch.abs(pred_cdf - target_cdf)
        
        if self.reduction == 'none':
            return wasserstein_dist
        elif self.reduction == 'sum':
            return wasserstein_dist.sum()
        else:  # Default: 'mean'
            return wasserstein_dist.mean()
        
class CombinedLoss(nn.Module):
    def __init__(
        self, 
        focal_weight=0.6, 
        wasserstein_weight=0.4, 
        focal_gamma=1.0, 
        focal_alpha=0.7,
        label_smoothing=0.1,  # Add label smoothing
        class_weights=None   # Optional class weights
    ):
        super().__init__()
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, 
            alpha=focal_alpha,
            label_smoothing=label_smoothing
        )
        self.wasserstein_loss = WassersteinLoss()
        self.focal_weight = focal_weight
        self.wasserstein_weight = wasserstein_weight
        
        # Add class weights if provided
        self.class_weights = class_weights
        
    def forward(self, inputs, targets):
        # Apply class weights if provided
        if self.class_weights is not None:
            inputs = inputs * self.class_weights.to(inputs.device)
        
        return (
            self.focal_weight * self.focal_loss(inputs, targets) + 
            self.wasserstein_weight * self.wasserstein_loss(inputs, targets)
        )
    
def apply_temperature_scaling(model, val_loader, device):
    """
    Apply temperature scaling to a trained model
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Temperature-scaled model
    """
    # Set model to eval mode
    model.eval()
    
    # Collect validation predictions
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)
    
    # Concatenate all batches
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Create temperature parameter - make sure it requires gradients
    temperature = nn.Parameter(torch.ones(1, requires_grad=True, device=device))
    
    # Define NLL loss criterion
    nll_criterion = nn.CrossEntropyLoss()
    
    # Define optimizer for temperature parameter only
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    # Define optimization step
    def eval_step():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss
    
    # Run optimization
    optimizer.step(eval_step)
    
    # Print optimal temperature
    print(f"Optimal temperature: {temperature.item():.4f}")
    
    # Create scaled model
    class TemperatureScaledModel(nn.Module):
        def __init__(self, model, temperature):
            super().__init__()
            self.model = model
            self.temperature = temperature
        
        def forward(self, x):
            logits = self.model(x)
            return logits / self.temperature
    
    # Return the calibrated model
    return TemperatureScaledModel(model, temperature.item())
    
class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        # Create ordinal encoding
        ordinal_targets = F.one_hot(targets, num_classes=self.num_classes).float()
        ordinal_targets = torch.cumsum(ordinal_targets, dim=1)
        
        # Compute probability of cumulative distribution
        probs = torch.sigmoid(inputs)
        
        # Compute binary cross-entropy for ordinal targets
        loss = F.binary_cross_entropy(probs, ordinal_targets)
        
        return loss
    
class OrdinalEarthMoversDistanceLoss(nn.Module):
    def __init__(self, num_classes=10, alpha=0.7, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Weight balance between CE and EMD
        self.label_smoothing = label_smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def forward(self, inputs, targets):
        # Standard cross-entropy with label smoothing
        ce = self.ce_loss(inputs, targets)
        
        # Earth Mover's Distance component
        probs = F.softmax(inputs, dim=1)
        
        # Create target distribution with ordinal smoothing
        target_dist = torch.zeros_like(probs)
        for i in range(len(targets)):
            target_class = targets[i].item()
            for c in range(self.num_classes):
                # Distance-based smoothing - closer classes get higher probability
                distance = abs(c - target_class)
                if distance == 0:
                    target_dist[i, c] = 1.0 - self.label_smoothing
                else:
                    # Other classes get probability that decreases with distance
                    target_dist[i, c] = self.label_smoothing * (1 / (1 + distance)) / (self.num_classes - 1)
            
            # Normalize
            target_dist[i] = target_dist[i] / target_dist[i].sum()
        
        # Calculate cumulative distributions
        target_cdf = torch.cumsum(target_dist, dim=1)
        pred_cdf = torch.cumsum(probs, dim=1)
        
        # Earth Mover's Distance (L1 distance between CDFs)
        emd = torch.abs(pred_cdf - target_cdf).sum(dim=1).mean()
        
        # Combined loss
        return self.alpha * ce + (1 - self.alpha) * emd
    
class SkinToneOrdinalLoss(nn.Module):
    def __init__(self, num_classes=7, alpha=0.7, beta=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for cross-entropy
        self.beta = beta    # Weight for ordinal penalty
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, outputs, targets):
        # Cross-entropy loss component
        ce = self.ce_loss(outputs, targets)
        
        # Ordinal penalty component
        batch_size = outputs.size(0)
        device = outputs.device
        
        # Calculate ordinal loss
        ordinal_loss = torch.tensor(0.0, device=device)
        
        # Convert outputs to probabilities
        probs = F.softmax(outputs, dim=1)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # Calculate weighted distance penalty
            for j in range(self.num_classes):
                if j != true_class:
                    # Penalty increases with distance from true class
                    distance = abs(j - true_class)
                    ordinal_loss += distance * probs[i, j]
        
        # Normalize
        ordinal_loss = ordinal_loss / batch_size
        
        # Combine losses with weights
        return self.alpha * ce + self.beta * ordinal_loss
    
    def label_smoothing_ordinal(self, targets, num_classes, smoothing=0.1):
        """
        Create smoothed labels that respect ordinal relationships.
        Classes closer to the target get more probability mass.
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Initialize smoothed labels
        smoothed_labels = torch.zeros(batch_size, num_classes, device=device)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # Assign highest probability to true class
            smoothed_labels[i, true_class] = 1.0 - smoothing
            
            # Distribute remaining probability based on distance
            remaining_prob = smoothing
            total_distance = 0
            
            # Calculate total distance for normalization
            for j in range(num_classes):
                if j != true_class:
                    # Inverse distance (closer classes get more weight)
                    total_distance += 1.0 / (1.0 + abs(j - true_class))
            
            # Distribute probability
            for j in range(num_classes):
                if j != true_class:
                    # Inverse distance weighting
                    distance_weight = 1.0 / (1.0 + abs(j - true_class))
                    smoothed_labels[i, j] = remaining_prob * (distance_weight / total_distance)
        
        return smoothed_labels

class FocalOrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=7, alpha=0.25, gamma=2.0, beta=0.3, 
                 wasserstein_weight=0.4, smoothing=0.1, 
                 contrastive_margin=0.3, contrastive_weight=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Focal loss parameter
        self.gamma = gamma  # Focal loss parameter
        self.beta = beta    # Weight for ordinal component
        self.wasserstein_weight = wasserstein_weight  # Weight for Wasserstein component
        self.smoothing = smoothing  # Label smoothing factor
        self.contrastive_margin = contrastive_margin  # Margin for adjacent class separation
        self.contrastive_weight = contrastive_weight  # Weight for contrastive loss
        
        # Class weights to counter severe prediction bias toward classes 3-4
        # Significantly boosting classes 1, 2, 5, 6, 7 which are underrepresented
        self.class_weights = torch.tensor([1.5, 2.5, 1.0, 0.5, 2.5, 3.5, 6.0])

        #self.class_weights = torch.tensor([2.5, 1.0, 1.0, 1.0, 1.5, 2.5, 3.0])
        # Pre-compute class weights to focus on extremes
        #self.class_weights = torch.tensor([1.1, 0.7, 0.8, 0.7, 1.5, 1.3, 3.5])
        #self.class_weights = torch.tensor([2.0, 2.0, 0.8, 1.2, 1.5, 2.5, 3.5])
        #self.class_weights = torch.tensor([1.5, 5.0, 0.5, 0.5, 1.5, 2.5, 3.5])
        #self.class_weights = torch.tensor([1.5, 10.0, 0.5, 0.5, 1.5, 2.5, 3.5])
        
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        device = inputs.device
        self.class_weights = self.class_weights.to(device)
        
        # Apply class weights
        weighted_inputs = inputs.clone()
        for c in range(self.num_classes):
            weighted_inputs[:, c] = weighted_inputs[:, c] * self.class_weights[c]
        
        # Compute standard cross-entropy with class weighting
        ce_loss = F.cross_entropy(weighted_inputs, targets)
        
        # Focal component - focus training on hard examples
        probs = F.softmax(inputs, dim=1)
        pt = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (self.alpha * focal_weight * ce_loss).mean()
        
        # Wasserstein Loss component - specifically helps with ordinal relationships
        pred_probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Calculate cumulative distributions
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(targets_one_hot, dim=1)
        
        # Earth Mover's Distance
        wasserstein_loss = torch.abs(pred_cdf - target_cdf).sum(dim=1).mean()
        
        # Special Class 1-2 separation component (85.3% confusion rate)
        class12_loss = torch.tensor(0.0, device=device)
        class1_mask = (targets == 0)  # Class 1 (0-indexed)
        class2_mask = (targets == 1)  # Class 2 (0-indexed)
        
        if torch.any(class1_mask):
            # For Class 1 samples, penalize if Class 2 probability is too high
            class1_inputs = inputs[class1_mask]
            class1_probs = F.softmax(class1_inputs, dim=1)
            class12_diff = class1_probs[:, 0] - class1_probs[:, 1]  # Class 1 prob - Class 2 prob
            class12_loss += torch.mean(F.relu(0.4 - class12_diff))  # Want at least 0.4 difference
            
        if torch.any(class2_mask):
            # For Class 2 samples, penalize if Class 3 probability is too high
            class2_inputs = inputs[class2_mask]
            class2_probs = F.softmax(class2_inputs, dim=1)
            class23_diff = class2_probs[:, 1] - class2_probs[:, 2]  # Class 2 prob - Class 3 prob
            class12_loss += torch.mean(F.relu(0.4 - class23_diff))  # Want at least 0.4 difference
        
        # Special penalty for class 3-4 confusion
        class34_loss = torch.tensor(0.0, device=device)
        class3_mask = (targets == 2)  # Class 3 (0-indexed)
        class4_mask = (targets == 3)  # Class 4 (0-indexed)
        
        if torch.any(class3_mask):
            class3_inputs = inputs[class3_mask]
            class3_probs = F.softmax(class3_inputs, dim=1)
            class34_diff = class3_probs[:, 2] - class3_probs[:, 3]
            class34_loss += torch.mean(F.relu(0.3 - class34_diff))
            
        if torch.any(class4_mask):
            class4_inputs = inputs[class4_mask]
            class4_probs = F.softmax(class4_inputs, dim=1)
            class43_diff = class4_probs[:, 3] - class4_probs[:, 2]
            class34_loss += torch.mean(F.relu(0.3 - class43_diff))
        
        # Add calibration penalty to reduce overconfidence
        calibration_loss = torch.tensor(0.0, device=device)
        confidence, predictions = torch.max(probs, dim=1)
        correct_prediction_mask = (predictions == targets)
        
        # Penalize high confidence when prediction is wrong
        if torch.any(~correct_prediction_mask):
            wrong_confidences = confidence[~correct_prediction_mask]
            calibration_loss = torch.mean(wrong_confidences)
        
        # Combine all losses
        total_loss = focal_loss + \
                     self.wasserstein_weight * wasserstein_loss + \
                     0.3 * class12_loss + \
                     0.3 * class34_loss + \
                     0.2 * calibration_loss
        
        return total_loss
    
class FocalOrdinalWassersteinLoss(nn.Module):
    def __init__(self, num_classes=7, alpha=0.25, gamma=2.0, 
                 ordinal_weight=0.3, wasserstein_weight=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Focal loss parameter
        self.gamma = gamma  # Focal loss parameter
        self.ordinal_weight = ordinal_weight  # Weight for ordinal component
        self.wasserstein_weight = wasserstein_weight  # Weight for Wasserstein component
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Base loss with smoothing
        
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Basic cross-entropy with label smoothing
        ce_loss = self.ce_loss(inputs, targets)  # This is already a scalar
        
        # Focal component - focus training on hard examples
        probs = F.softmax(inputs, dim=1)
        pt = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = (self.alpha * focal_weight * ce_loss).mean()  # Ensure scalar output
        
        # Explicit ordinal penalty - penalize predictions based on distance
        ordinal_loss = torch.tensor(0.0, device=device)  # Initialize as scalar tensor
        for i in range(batch_size):
            true_class = targets[i].item()
            for j in range(self.num_classes):
                if j != true_class:
                    # Penalize based on distance between classes
                    distance = abs(j - true_class)
                    ordinal_loss += distance * probs[i, j]
        
        ordinal_loss = ordinal_loss / batch_size  # This is a scalar
        
        # Wasserstein Loss component for ordinal relationships
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Calculate cumulative distributions
        pred_cdf = torch.cumsum(probs, dim=1)
        target_cdf = torch.cumsum(targets_one_hot, dim=1)
        
        # Earth Mover's Distance
        wasserstein_loss = torch.abs(pred_cdf - target_cdf).sum(dim=1).mean()  # This is a scalar
        
        # Combined loss with all three components
        total_loss = focal_loss + self.ordinal_weight * ordinal_loss + self.wasserstein_weight * wasserstein_loss
        
        return total_loss  
    
class BalancedOrdinalLoss(nn.Module):
    def __init__(self, num_classes=7, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
        # More aggressive class weights based on confusion patterns
        self.class_weights = torch.tensor([0.8, 1.2, 6.0, 1.0, 0.8, 3.0, 3.0])
        
    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        device = inputs.device
        self.class_weights = self.class_weights.to(device)
        
        # Apply class weights
        weighted_inputs = inputs.clone()
        for c in range(self.num_classes):
            weighted_inputs[:, c] = weighted_inputs[:, c] * self.class_weights[c]
        
        # 1. Base cross-entropy loss with label smoothing
        log_probs = F.log_softmax(weighted_inputs, dim=1)
        
        # Create smoothed labels with ordinal properties
        smoothed_targets = torch.zeros(batch_size, self.num_classes, device=device)
        for i in range(batch_size):
            true_class = targets[i].item()
            for c in range(self.num_classes):
                if c == true_class:
                    smoothed_targets[i, c] = 1.0 - self.smoothing
                else:
                    # Distribute smoothing based on distance
                    distance = abs(c - true_class)
                    smoothed_targets[i, c] = self.smoothing * (1.0 / (1.0 + distance**2))
            # Normalize
            smoothed_targets[i] /= smoothed_targets[i].sum()
        
        # Compute cross-entropy with smoothed labels
        ce_loss = -(smoothed_targets * log_probs).sum(dim=1).mean()
        
        # 2. Class-specific penalties based on confusion patterns
        class_penalties = torch.zeros(1, device=device)
        probs = F.softmax(inputs, dim=1)
        
        # Class 3 penalty (severely underpredicted)
        class3_mask = (targets == 2)  # Class 3 is index 2
        if torch.any(class3_mask):
            class3_probs = probs[class3_mask, 2]  # Class 3 probabilities
            # Penalty for low probability on the correct class
            class3_penalty = (1.0 - class3_probs).mean()
            class_penalties += 2.0 * class3_penalty  # Very strong weight
            
        # Class 6 penalty (also underpredicted)
        class6_mask = (targets == 5)  # Class 6 is index 5
        if torch.any(class6_mask):
            class6_probs = probs[class6_mask, 5]  # Class 6 probabilities
            class6_penalty = (1.0 - class6_probs).mean()
            class_penalties += 1.5 * class6_penalty
            
        # Class 1-2 separation (too much confusion)
        class12_confusion = torch.zeros(1, device=device)
        
        class1_mask = (targets == 0)  # Class 1 is index 0
        if torch.any(class1_mask):
            # For class 1, penalize high probability on class 2
            class1_on_class2_probs = probs[class1_mask, 1]  # Class 2 probabilities
            class12_confusion += class1_on_class2_probs.mean()
            
        class2_mask = (targets == 1)  # Class 2 is index 1
        if torch.any(class2_mask):
            # For class 2, penalize high probability on class 1
            class2_on_class1_probs = probs[class2_mask, 0]  # Class 1 probabilities
            class12_confusion += class2_on_class1_probs.mean()
        
        # 3. Ordinal relationship enforcement
        ordinal_penalty = torch.zeros(1, device=device)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # Progressively penalize more for further classes
            for c in range(self.num_classes):
                if c != true_class:
                    distance = abs(c - true_class)
                    ordinal_penalty += distance * probs[i, c]
                    
        ordinal_penalty = ordinal_penalty / batch_size
        
        # 4. Margin between adjacent classes
        margin_loss = torch.zeros(1, device=device)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # Get probability of true class
            true_prob = probs[i, true_class]
            
            # For adjacent classes, enforce a margin
            for adj_class in [true_class-1, true_class+1]:
                if 0 <= adj_class < self.num_classes:
                    adj_prob = probs[i, adj_class]
                    margin = 0.3  # Desired margin
                    margin_loss += F.relu(adj_prob - (true_prob - margin))
        
        margin_loss = margin_loss / batch_size
        
        # 5. Entropy regularization to prevent overconfident predictions
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
        entropy_penalty = -0.1 * entropy  # Small weight to encourage some uncertainty
        
        # Combine all components
        total_loss = ce_loss + 0.5 * ordinal_penalty + 0.7 * class_penalties + 0.5 * class12_confusion + 0.3 * margin_loss + entropy_penalty
        
        return total_loss
    
class EnhancedOrdinalLoss(nn.Module):
    def __init__(self, num_classes=7, alpha=0.7, gamma=2.0):
        super().__init__()
        self.base_loss = OrdinalEarthMoversDistanceLoss(num_classes, alpha)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        base_loss = self.base_loss(inputs, targets)
        
        # Add focal component for middle classes
        probs = F.softmax(inputs, dim=1)
        
        # Focus on class 3 (index 2) which has the lowest ROC AUC
        class3_mask = (targets == 2)
        if class3_mask.sum() > 0:
            class3_probs = probs[class3_mask, 2]
            focal_weight = (1 - class3_probs) ** self.gamma
            class3_focal = focal_weight.mean()
            
            # Also add some focus on class 1
            class1_mask = (targets == 0)
            if class1_mask.sum() > 0:
                class1_probs = probs[class1_mask, 0]
                class1_focal = ((1 - class1_probs) ** self.gamma).mean()
                return base_loss + 0.2 * class3_focal + 0.15 * class1_focal
            
            return base_loss + 0.2 * class3_focal
        
        return base_loss
    
class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0, num_classes=7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        # Standard Cross Entropy
        ce_loss = F.cross_entropy(inputs, targets)
        
        # Reverse Cross Entropy
        pred_prob = F.softmax(inputs, dim=1)
        rev_targets = F.one_hot(targets, num_classes=self.num_classes).float()
        rev_ce_loss = -torch.mean(torch.sum(pred_prob * rev_targets, dim=1))
        
        # Combined loss
        return self.alpha * ce_loss + self.beta * rev_ce_loss
    
class WeightedLabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1, class_weights=None):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        
        # Convert pandas Series to tensor and move to correct device when needed
        if class_weights is not None:
            if hasattr(class_weights, 'values'):  # If it's a pandas Series
                self.class_weights = torch.tensor(class_weights.values, dtype=torch.float32)
            else:
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
    
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        loss = (1 - self.smoothing) * F.nll_loss(log_prob, targets) + \
               self.smoothing * (-log_prob.mean(dim=-1))
        
        # Add class weights if provided
        if self.class_weights is not None:
            # Move weights to the same device as targets
            device_weights = self.class_weights.to(targets.device)
            # Get weights for each target in the batch
            weights_batch = device_weights[targets]
            loss = loss * weights_batch
        
        return loss.mean()
    
class AdvancedWassersteinLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        # Compute cumulative distribution
        pred_probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        pred_cdf = torch.cumsum(pred_probs, dim=1)
        target_cdf = torch.cumsum(targets_one_hot, dim=1)
        
        # Compute Wasserstein distance with squared error
        wasserstein_dist = torch.mean(torch.square(pred_cdf - target_cdf))
        
        return wasserstein_dist
    
def distance_weighted_accuracy(y_true, y_pred, max_distance=9):
    """
    Calculate accuracy weighted by the distance between classes.
    Predictions that are closer to the true class receive higher scores.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        max_distance: Maximum possible distance between classes
    
    Returns:
        Distance-weighted accuracy score
    """
    distances = np.abs(y_pred - y_true)
    weights = 1.0 - (distances / max_distance)  # Higher weight for smaller distances
    return np.mean(weights)

def off_by_one_accuracy(y_true, y_pred):
    """
    Calculate the percentage of predictions that are at most one class 
    away from the true label.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
    
    Returns:
        Off-by-one accuracy score
    """
    return np.mean(np.abs(y_pred - y_true) <= 1)

def class_pair_confusion_rate(y_true, y_pred, class_a, class_b):
    """
    Calculate the confusion rate between a specific pair of classes.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_a: First class index
        class_b: Second class index
    
    Returns:
        Confusion rate between the specified classes
    """
    # Filter samples where true label is either class_a or class_b
    mask = (y_true == class_a) | (y_true == class_b)
    if not np.any(mask):
        return 0.0
    
    # Get subset of true and predicted labels
    true_subset = y_true[mask]
    pred_subset = y_pred[mask]
    
    # Calculate confusion rate: samples where true and predicted classes don't match
    return np.mean(true_subset != pred_subset)

def calculate_similar_classes_confusion(y_true, y_pred, similar_classes=[0, 1, 2, 3]):
    """
    Calculate confusion rates between all pairs of similar classes.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        similar_classes: List of class indices considered similar
    
    Returns:
        Dictionary of confusion rates for each class pair
    """
    confusion_rates = {}
    for i in similar_classes:
        for j in similar_classes:
            if i != j:
                rate = class_pair_confusion_rate(y_true, y_pred, i, j)
                confusion_rates[f"{i+1}_vs_{j+1}"] = rate
    return confusion_rates

def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: predicted probabilities (output of softmax) - shape (n_samples, n_classes)
        labels: ground truth labels - shape (n_samples,)
        n_bins: number of bins for confidence scores
    
    Returns:
        Expected Calibration Error score
    """
    # Get the confidence and predictions
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate ECE across bins
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Calculate accuracy and average confidence in the bin
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            
            # Add weighted absolute difference to ECE
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
    
    return ece

def compute_brier_score(probs, labels, n_classes):
    """
    Compute Brier Score for multiclass classification.
    
    Args:
        probs: predicted probabilities (output of softmax) - shape (n_samples, n_classes)
        labels: ground truth labels - shape (n_samples,)
        n_classes: number of classes
    
    Returns:
        Brier Score
    """
    # One-hot encode the labels
    y_true_one_hot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        y_true_one_hot[i, label] = 1
    
    # Calculate squared error between predictions and true one-hot vectors
    return np.mean(np.sum((probs - y_true_one_hot) ** 2, axis=1))

def plot_calibration_curve(probs, labels, output_dir=None, n_bins=10):
    """
    Plot a calibration curve to visualize model calibration.
    
    Args:
        probs: predicted probabilities (output of softmax)
        labels: ground truth labels
        output_dir: directory to save the plot
        n_bins: number of bins for confidence scores
    """

    os.makedirs(output_dir, exist_ok=True)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_counts.append(np.sum(in_bin))
        
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    
    # Add bin counts as bar chart in background
    ax2 = plt.gca().twinx()
    ax2.bar(bin_lowers + (bin_uppers - bin_lowers) / 2, bin_counts, 
            width=(bin_uppers - bin_lowers) * 0.8, alpha=0.3, color='gray', label='Samples per bin')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    plt.close()

def save_training_parameters(optimizer, criterion, scheduler, output_dir=None):
    """
    Save details about the optimizer, criterion, and scheduler used for training.
    
    Args:
        optimizer: The optimizer used for training
        criterion: The loss function used for training
        scheduler: The learning rate scheduler used
        output_dir: Directory to save the file (defaults to config.OUTPUT_DIR)
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"training_parameters_{timestamp}.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"=== Training Parameters ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Save criterion details
        f.write("=== Loss Function ===\n")
        f.write(f"Type: {criterion.__class__.__name__}\n")
        if isinstance(criterion, SymmetricCrossEntropyLoss):
            f.write(f"Alpha: {criterion.alpha}\n")
            f.write(f"Beta: {criterion.beta}\n")
            f.write(f"Number of Classes: {criterion.num_classes}\n")
        
        elif isinstance(criterion, WeightedLabelSmoothingLoss):
            f.write(f"Smoothing: {criterion.smoothing}\n")
            f.write(f"Number of Classes: {criterion.num_classes}\n")
            f.write(f"Class Weights: {'Present' if criterion.class_weights is not None else 'None'}\n")
        
        elif isinstance(criterion, OrdinalRegressionLoss):
            f.write(f"Number of Classes: {criterion.num_classes}\n")
        
        elif isinstance(criterion, AdvancedWassersteinLoss):
            f.write(f"Number of Classes: {criterion.num_classes}\n")

        elif isinstance(criterion, OrdinalEarthMoversDistanceLoss):
            f.write(f"Number of Classes: {criterion.num_classes}\n")
            f.write(f"Alpha: {criterion.alpha}\n")
            f.write(f"Label Smoothing: {criterion.label_smoothing}\n")

        elif isinstance(criterion, EnhancedOrdinalLoss):
            f.write(f"Gamma: {criterion.gamma}\n")

        elif isinstance(criterion, SkinToneOrdinalLoss):
            f.write(f"Beta: {criterion.beta}\n")
            f.write(f"Alpha: {criterion.alpha}\n")

        elif isinstance(criterion, FocalOrdinalCrossEntropyLoss):
            f.write(f"Beta: {criterion.beta}\n")
            f.write(f"Alpha: {criterion.alpha}\n")
            f.write(f"Gamma: {criterion.gamma}\n")      
            
        elif isinstance(criterion, FocalLoss):
            f.write(f"Alpha: {criterion.alpha}\n")
            f.write(f"Gamma: {criterion.gamma}\n")
            f.write(f"Reduction: {criterion.reduction}\n")
        elif isinstance(criterion, nn.CrossEntropyLoss):
            f.write(f"Weight: {criterion.weight}\n" if hasattr(criterion, 'weight') and criterion.weight is not None else "Weight: None\n")
            f.write(f"Label Smoothing: {criterion.label_smoothing}\n" if hasattr(criterion, 'label_smoothing') else "Label Smoothing: 0.0\n")
        elif isinstance(criterion, CombinedLoss):
            f.write(f"focal_weight: {criterion.focal_weight}\n")
            f.write(f"wasserstein_weight: {criterion.wasserstein_weight}\n")
            f.write(f"Alpha: {criterion.focal_loss.alpha}\n")
            f.write(f"Gamma: {criterion.focal_loss.gamma}\n")
            f.write(f"label_smoothing: {criterion.focal_loss.label_smoothing}\n")
        f.write("\n")
        
        # Save optimizer details
        f.write("=== Optimizer ===\n")
        f.write(f"Type: {optimizer.__class__.__name__}\n")
        f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Weight Decay: {optimizer.param_groups[0]['weight_decay']}\n")
        if isinstance(optimizer, optim.SGD):
            f.write(f"Momentum: {optimizer.param_groups[0]['momentum']}\n")
            f.write(f"Nesterov: {optimizer.param_groups[0]['nesterov']}\n")
        f.write("\n")
        
        # Save scheduler details
        f.write("=== Learning Rate Scheduler ===\n")
        if scheduler is None:
            f.write("No scheduler used\n")
        else:
            scheduler_type = scheduler.__class__.__name__
            f.write(f"Type: {scheduler_type}\n")

            if scheduler_type == 'ReduceLROnPlateau':
                f.write(f"Mode: {scheduler.mode}\n")
                f.write(f"Factor: {scheduler.factor}\n")
                f.write(f"Patience: {scheduler.patience}\n")
                f.write(f"Threshold: {scheduler.threshold}\n")
                f.write(f"Cooldown: {scheduler.cooldown}\n")

            elif scheduler_type == 'OneCycleLR':
                f.write(f"\n")

            elif scheduler_type == 'CosineAnnealingWarmRestarts':
                f.write(f"T_0: {scheduler.T_0}\n")
                f.write(f"T_mult: {scheduler.T_mult}\n")
                f.write(f"eta_min: {scheduler.eta_min}\n")
                
    print(f"Training parameters saved to {file_path}")
    return file_path

# Main function
def main(config):
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Using model type: {config.MODEL_TYPE}")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader, monk_scale_counts, class_weights = prepare_data(
    csv_path=config.CSV_PATH,
    image_dir=config.IMAGE_DIR,
    batch_size=config.BATCH_SIZE,
    test_batch_size=config.TEST_BATCH_SIZE,
    val_size=config.VAL_SIZE,
    test_size=config.TEST_SIZE,
    augmentation_method="both" 
)

    #analyze_data_distribution(train_loader, val_loader, test_loader, 
    #                          num_classes=config.NUM_CLASSES, 
    #                          output_dir=config.OUTPUT_DIR)
    
    
    # Create model based on model type
    
    if config.MODEL_TYPE == 'b0':
        # Use the new attention model instead
        model = AttentionSkinToneClassifier(
            num_classes=config.NUM_CLASSES,
            model_type='b0',
            freeze_features=config.FREEZE_FEATURES
        )
    elif config.MODEL_TYPE == 'b3':
        model = AttentionSkinToneClassifier(
            num_classes=config.NUM_CLASSES,
            model_type='b3', 
            freeze_features=config.FREEZE_FEATURES
        )
    elif config.MODEL_TYPE == 'b5':
        model = AttentionSkinToneClassifier(
            num_classes=config.NUM_CLASSES,
            model_type='b5',
            freeze_features=config.FREEZE_FEATURES
        )
    
    #model = TemperatureScaledModel(model)
    #model = SkinToneFusionModel(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    # OPTION 1: FocalLoss
    #criterion = FocalLoss(gamma=2.0, alpha=0.25)
    
    # OPTION 2: Weighted CrossEntropy with label smoothing
    # class_weights = get_class_weights(monk_scale_counts).to(config.DEVICE)
    # criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Option 3: Use Wasserstein loss alone
    #criterion = WassersteinLoss()

    #class_weights = get_class_weights(monk_scale_counts)

    # Option 4: Use combined loss
    #criterion = CombinedLoss(
    #        focal_weight=0.5, 
    #        wasserstein_weight=0.5,
    #        focal_gamma=0.5, 
    #        focal_alpha=0.5,
    #        label_smoothing=0.1,
    #        #class_weights=class_weights
    #    ) 
    
    # Option 5: Symmetric Cross Entropy Loss
    #criterion = SymmetricCrossEntropyLoss(
    #    alpha=0.1,  # Adjust these parameters
    #    beta=1.0,
    #    num_classes=config.NUM_CLASSES
    #)

    # Option 6: Weighted Label Smoothing Loss
    #criterion = WeightedLabelSmoothingLoss(
    #    num_classes=config.NUM_CLASSES, 
    #    smoothing=0.1,
    #    class_weights=class_weights
    #)

    # Option 7: Ordinal Regression Loss
    #criterion = OrdinalRegressionLoss(
    #    num_classes=config.NUM_CLASSES
    #)

    # Option 8: Advanced Wasserstein Loss
    #criterion = AdvancedWassersteinLoss(
    #    num_classes=config.NUM_CLASSES
    #)

    # Option 9: Ordinal Earth Movers Distance Loss
    #criterion =OrdinalEarthMoversDistanceLoss(
    #    num_classes = config.NUM_CLASSES, 
    #    alpha=0.7, 
    #    label_smoothing=0.1
    #)

    # Option 10: Enhanced Ordinal Distance Loss
    #criterion = EnhancedOrdinalLoss(
    #    num_classes=7, 
    #    alpha=0.7, 
    #    gamma=2.0
    #)

    #criterion = SkinToneOrdinalLoss()

    #trainable_params = [p for p in model.parameters() if p.requires_grad]

    #criterion = FocalOrdinalCrossEntropyLoss()
    #criterion = BalancedOrdinalLoss()

    criterion = FocalOrdinalWassersteinLoss()
    
    # OPTION 1: AdamW Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0003,  
        weight_decay=0.0001
    )
    
    # OPTION 2: SGD with Momentum
    #optimizer = optim.SGD(
    #    model.parameters(),
    #    lr=0.0001,
    #    momentum=0.9,
    #    nesterov=True,
    #    weight_decay=0.01
    #)

    # Option 3: RAdam (Rectified Adam)
    #optimizer = optim.RAdam(
    #    model.parameters(),
    #    lr=0.002,
    #    weight_decay=1e-5
    #)

    # Option 4: AdamAW (Adaptive Adam with Warmup)
    #optimizer = optim.AdamW(
    #    model.parameters(),
    #    lr=0.0005,  # Very small initial learning rate
    #    betas=(0.9, 0.999),
    #    weight_decay=1e-4,
    #    amsgrad=True
    #)
    
   # Option 1: CosineAnnealingWarmRestarts (Current approach)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,     # Initial restart period
        T_mult=2,  # Multiplicative factor for restarts
        eta_min=1e-6
    )

    # Option 2: OneCycleLR with warmup and annealing
    #scheduler = optim.lr_scheduler.OneCycleLR(
    #    optimizer,
    #    max_lr=0.001,
    #    steps_per_epoch=len(train_loader),
    #    epochs=100,  
    #    pct_start=0.1,  
    #    div_factor=10,
    #    final_div_factor=100
    #)

    # Option 3: Reduce LR on Plateau with custom parameters
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer,
    #    mode='min',
    #    factor=0.5,
    #    patience=5,
    #    verbose=True,
    #    min_lr=1e-6
    #)

    # Option 4: ExponentialLR with periodic restarts
    #scheduler = optim.lr_scheduler.ExponentialLR(
    #    optimizer, 
    #    gamma=0.9,      # Exponential decay
    #    last_epoch=-1
    #)

    # Option 5: Cyclic Learning Rate
    #scheduler = optim.lr_scheduler.CyclicLR(
    #    optimizer,
    #    base_lr=0.001,  # Lower base learning rate
    #    max_lr=0.01,    # Higher peak learning rate
    #    step_size_up=5,  # Number of iterations for increasing LR
    #    mode='triangular2',  # Triangular policy with amplitude reduction
    #    cycle_momentum=True
    #)

    architecture_file = save_model_architecture(
        model, 
        model_name=f"EfficientNet{config.MODEL_TYPE}", 
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

    # Apply temperature scaling for calibration
    print("Calibrating model with temperature scaling...")
    calibrated_model = apply_temperature_scaling(model, val_loader, config.DEVICE)
    
    # Save the calibrated model
    torch.save({
        'model_state_dict': model.state_dict(),
        'temperature': calibrated_model.temperature,
    }, os.path.join(config.OUTPUT_DIR, 'calibrated_model.pth'))
    
    # Evaluate original model first
    print("Evaluating original model...")
    original_results = evaluate_model(
        model,
        test_loader,
        device=config.DEVICE,
        output_dir=os.path.join(config.OUTPUT_DIR, 'original_model_results')
    )
    
    # Unpack original results
    (orig_acc, orig_balanced_acc, orig_precision_macro, orig_recall_macro, orig_f1_macro, 
     orig_mae, orig_mse, orig_kappa, orig_roc_auc, orig_dw_acc, orig_off_by_one_acc, 
     orig_spearman_corr, orig_ece, orig_brier_score, orig_cm) = original_results
    
    # Save original model metrics
    orig_experiment_metrics = {
        'model_type': config.MODEL_TYPE,
        'freeze_features': config.FREEZE_FEATURES,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.NUM_EPOCHS,
        'test_accuracy': orig_acc,
        'test_balanced_accuracy': orig_balanced_acc,
        'test_f1_macro': orig_f1_macro,
        'precision_macro': orig_precision_macro,
        'recall_macro': orig_recall_macro,
        'mean_absolute_error': orig_mae,
        'mean_square_error': orig_mse,
        'cohens_kappa': orig_kappa,
        'roc_auc_macro': orig_roc_auc,
        'distance_weighted_accuracy': orig_dw_acc,
        'off_by_one_accuracy': orig_off_by_one_acc,
        'spearman_correlation': orig_spearman_corr,
        'expected_calibration_error': orig_ece,
        'brier_score': orig_brier_score
    }
    
    orig_summary_file = save_metrics_to_file(
        orig_experiment_metrics,
        model_name=f"OriginalModel_{config.MODEL_TYPE}", 
        output_dir=os.path.join(config.OUTPUT_DIR, 'original_model_results'),
        test_size=len(test_loader.dataset)
    )
    
    # Then evaluate the temperature-scaled model
    print("Evaluating calibrated model...")
    calibrated_results = evaluate_model(
        calibrated_model,
        test_loader,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR  # Main output directory for the calibrated results
    )
    
    # Unpack calibrated results
    (cal_acc, cal_balanced_acc, cal_precision_macro, cal_recall_macro, cal_f1_macro, 
     cal_mae, cal_mse, cal_kappa, cal_roc_auc, cal_dw_acc, cal_off_by_one_acc, 
     cal_spearman_corr, cal_ece, cal_brier_score, cal_cm) = calibrated_results
    
    # Save calibrated model metrics - using your existing format
    experiment_metrics = {
        'model_type': config.MODEL_TYPE,
        'freeze_features': config.FREEZE_FEATURES,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'epochs': config.NUM_EPOCHS,
        'temperature': calibrated_model.temperature,
        'test_accuracy': cal_acc,
        'test_balanced_accuracy': cal_balanced_acc,
        'test_f1_macro': cal_f1_macro,
        'precision_macro': cal_precision_macro,
        'recall_macro': cal_recall_macro,
        'mean_absolute_error': cal_mae,
        'mean_square_error': cal_mse,
        'cohens_kappa': cal_kappa,
        'roc_auc_macro': cal_roc_auc,
        'distance_weighted_accuracy': cal_dw_acc,
        'off_by_one_accuracy': cal_off_by_one_acc,
        'spearman_correlation': cal_spearman_corr,
        'expected_calibration_error': cal_ece,
        'brier_score': cal_brier_score
    }
    
    # Save calibrated model metrics
    summary_file = save_metrics_to_file(
        experiment_metrics,
        model_name=f"CalibratedModel_{config.MODEL_TYPE}", 
        output_dir=config.OUTPUT_DIR,
        test_size=len(test_loader.dataset)
    )
    
    # Compare results
    print("\nComparison of models:")
    print(f"Original model - Accuracy: {orig_acc:.4f}, Balanced Acc: {orig_balanced_acc:.4f}, ECE: {orig_ece:.4f}")
    print(f"Calibrated model - Accuracy: {cal_acc:.4f}, Balanced Acc: {cal_balanced_acc:.4f}, ECE: {cal_ece:.4f}")
    
    # Save training parameters
    save_training_parameters(optimizer, criterion, None, output_dir=config.OUTPUT_DIR)
    
    print("Training and evaluation complete!")
    
    # Return the better model based on balanced accuracy
    if cal_balanced_acc > orig_balanced_acc:
        print(f"Calibrated model performs better and is saved to {os.path.join(config.OUTPUT_DIR, 'calibrated_model.pth')}")
        return calibrated_model, history
    else:
        print(f"Original model performs better and is saved to {os.path.join(config.OUTPUT_DIR, 'best_model.pth')}")
        return model, history


if __name__ == "__main__":
    config = ConfigB3()
    main(config)