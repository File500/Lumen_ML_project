import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

def get_random_transform():
    """
    Generate a random image transformation.
    
    Returns:
        transforms.Compose: A random transformation
    """
    possible_transforms = [
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
    ]
    return random.choice(possible_transforms)

def create_balanced_sampler(dataset, target_class, target_ratio):
    """
    Create a weighted sampler to balance class distribution.
    
    Args:
        dataset (torch.utils.data.Dataset): Input dataset
        target_class (int): Class to oversample
        target_ratio (float): Desired ratio of target class samples
    
    Returns:
        WeightedRandomSampler or None: Balanced sampler
    """
    # Get all labels from the dataset
    if hasattr(dataset, 'target'):
        labels = dataset.target
    else:
        return None

    # Calculate class counts
    class_count = np.bincount(labels)
    total_samples = len(labels)

    # Calculate the number of samples needed for target class
    target_samples = int(total_samples * target_ratio)
    current_target_samples = class_count[target_class]

    # Calculate weights for each sample
    weights = np.ones_like(labels, dtype=np.float64)

    # Calculate the weight adjustment needed for the target class
    if current_target_samples < target_samples:
        weights[labels == target_class] = target_samples / current_target_samples

    # Create and return the sampler
    return WeightedRandomSampler(weights, num_samples=total_samples, replacement=True)

def create_data_loader(data, batch_size, num_workers, sampler=None):
    """
    Create a DataLoader with specified configurations.
    
    Args:
        data (torch.utils.data.Dataset): Input dataset
        batch_size (int): Number of samples per batch
        num_workers (int): Number of subprocesses for data loading
        sampler (torch.utils.data.Sampler, optional): Sampling strategy
    
    Returns:
        torch.utils.data.DataLoader: Configured data loader
    """
    dataloader = DataLoader(
        data, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        sampler=sampler,
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=2
    )
    return dataloader

def stratified_split(full_df, train_size, val_size, test_size, random_state=42):
    """
    Perform stratified split based on both target and skin type.
    
    Args:
        full_df (pd.DataFrame): Full dataframe
        train_size (float): Proportion of training data
        val_size (float): Proportion of validation data
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: train_df, val_df, test_df
    """
    # Create a combined stratification column
    full_df['stratify_col'] = full_df['target'].astype(str) + '_' + full_df['monk_skin_type'].astype(str)
    
    # First split into train and temp (val+test combined) with stratification
    train_df, temp_df = train_test_split(
        full_df, 
        test_size=(val_size + test_size),
        stratify=full_df['stratify_col'],
        random_state=random_state
    )
    
    # Then split temp into val and test with stratification
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_size / (val_size + test_size)),
        stratify=temp_df['stratify_col'],
        random_state=random_state
    )
    
    # Remove the temporary stratification column
    train_df = train_df.drop('stratify_col', axis=1)
    val_df = val_df.drop('stratify_col', axis=1)
    test_df = test_df.drop('stratify_col', axis=1)
    
    return train_df, val_df, test_df