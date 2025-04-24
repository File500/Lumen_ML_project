import os
import torch
import pandas as pd
import numpy as np
import collections
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

class MonkScaleDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, cache_size=3000):
        """
        Dataset for Monk Scale images that filters out missing files.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.cache_size = cache_size
        
        # Filter out missing files
        valid_rows = []
        missing_count = 0
        
        for _, row in df.iterrows():
            image_name = row['image_name']
            if not image_name.endswith('.jpg'):
                image_name = f"{image_name}.jpg"
                
            img_path = os.path.join(image_dir, image_name)
            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                missing_count += 1
        
        # Create a new DataFrame with only valid rows
        self.df = pd.DataFrame(valid_rows)
        
        if missing_count > 0:
            print(f"Filtered out {missing_count} missing images. Dataset size: {len(self.df)}")
        
        # Initialize cache
        self.cache = collections.OrderedDict()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image name
        image_name = self.df.iloc[idx]['image_name']
        if not image_name.endswith('.jpg'):
            image_name = f"{image_name}.jpg"
        
        img_path = os.path.join(self.image_dir, image_name)
        
        # Check cache
        if img_path in self.cache:
            image = self.cache.pop(img_path)
            self.cache[img_path] = image
        else:
            # Load image (all images should exist at this point)
            image = Image.open(img_path).convert('RGB')
                
            # Add to cache
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[img_path] = image
        
        # Apply transformations
        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image
        
        # Get label
        label = self.df.iloc[idx]['monk_scale_type'] - 1
        
        return transformed_image, label


def get_transforms(model_type='b0'):
    """
    Create transformations for images.
    
    Args:
        model_type: 'b0', 'b3', or 'b5'
    """
    # Determine image size based on model type
    if model_type == 'b0':
        image_size = 224
    elif model_type == 'b3':
        image_size = 300
    elif model_type == 'b5':
        image_size = 456
    else:
        image_size = 224  # Default
    
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
            size=image_size,
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
        transforms.ToTensor(),
    ])

    return train_transform, val_transform

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

def prepare_data(
    csv_path, 
    image_dir, 
    batch_size=64,
    test_batch_size=64,
    val_size=0.15,
    test_size=0.15,
    seed=42,
    model_type='b3',
    augmentation_method="both",  # Options: "weighted_sampling", "duplicate_samples", "both"
    output_dir=None
):
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        csv_path: Path to CSV file with image names and labels
        image_dir: Directory containing the images
        batch_size: Batch size for training
        test_batch_size: Batch size for validation and testing
        val_size: Validation set size (proportion)
        test_size: Test set size (proportion)
        seed: Random seed for reproducibility
        model_type: Model type ('b0', 'b3', 'b5')
        augmentation_method: Augmentation method to use
        output_dir: Directory to save transformation details
        
    Returns:
        train_loader, val_loader, test_loader, monk_scale_counts, class_weights
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if image_name column exists, if not try to find a suitable column
    if 'image_name' not in df.columns:
        print("Warning: 'image_name' column not found in CSV.")
        if 'image_id' in df.columns:
            print("Using 'image_id' column as image name.")
            df['image_name'] = df['image_id']
        elif 'filename' in df.columns:
            print("Using 'filename' column as image name.")
            df['image_name'] = df['filename']
        else:
            potential_columns = [col for col in df.columns if 'image' in col.lower() or 'file' in col.lower()]
            if potential_columns:
                print(f"Using '{potential_columns[0]}' column as image name.")
                df['image_name'] = df[potential_columns[0]]
            else:
                raise ValueError("Could not find a suitable column for image names in the CSV.")
    
    # Check for NaN values in monk_scale_type
    nan_count = df['monk_scale_type'].isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} rows with NaN values in monk_scale_type. Dropping these rows.")
        df = df.dropna(subset=['monk_scale_type'])
    
    # Make sure monk_scale_type is numeric type
    df['monk_scale_type'] = pd.to_numeric(df['monk_scale_type'], errors='coerce')
    
    # Drop any rows with non-numeric values that were converted to NaN
    nan_after_conversion = df['monk_scale_type'].isna().sum()
    if nan_after_conversion > 0:
        print(f"Found {nan_after_conversion} rows with non-numeric values in monk_scale_type. Dropping these rows.")
        df = df.dropna(subset=['monk_scale_type'])
    
    # Verify the image directory exists
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    # Display dataset information
    print(f"Dataset shape after cleaning: {df.shape}")
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
    train_transform, val_transform = get_transforms(model_type)

    if output_dir:
        from utils import save_transformation_details
        save_transformation_details(train_transform, output_dir=output_dir)
    
    # Create datasets without pre-filtering
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

    
    # Update monk_scale_counts after filtering
    monk_scale_counts = train_dataset.df['monk_scale_type'].value_counts().sort_index()
    print("\nClass distribution after filtering missing files:")
    for scale, count in monk_scale_counts.items():
        print(f"  Monk Scale {scale}: {count} samples ({count/len(train_dataset.df)*100:.2f}%)")
    
    # Initialize class weights for potential use
    class_weights = None
    
    # Decide whether to use weighted sampling
    if augmentation_method == "weighted_sampling" or augmentation_method == "both":
        print("\nApplying weighted sampling strategy...")
        
        # Calculate inverse class weights with boosting for extreme classes
        class_weights = {}
        total_samples = len(train_dataset.df)
        
        # Adjust weights based on whether we're using both methods
        boost_factor_1 = 2.5 if augmentation_method == "weighted_sampling" else 1.5  # Reduced if using both
        boost_factor_7 = 3.0 if augmentation_method == "weighted_sampling" else 2.0  # Reduced if using both
        boost_factor_mid = 1.5 if augmentation_method == "weighted_sampling" else 1.2  # Reduced if using both
        
        # Calculate weights using the updated DataFrame
        monk_scale_counts = train_dataset.df['monk_scale_type'].value_counts().sort_index()
        
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
        
        # Apply weights to each sample using the updated DataFrame
        sample_weights = [class_weights[cls] for cls in train_dataset.df['monk_scale_type']]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset.df),
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