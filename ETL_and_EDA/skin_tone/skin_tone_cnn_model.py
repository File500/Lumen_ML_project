#!/usr/bin/env python3
"""
CNN model module for Monk skin type classification.
This module handles training and prediction using a CNN model.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import collections

# Set global variables
MONK_SKIN_TYPES = 10  # 10 levels from lightest (1) to darkest (10)

class CachedSkinDataset(Dataset):
    """Dataset for skin images with skin type labels and caching."""
    
    def __init__(self, dataframe, img_dir, transform=None, binary_mode=False, cache_images=True):
        """
        Initialize the dataset.
        
        Args:
            dataframe (pandas.DataFrame): DataFrame containing image info and labels
            img_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            binary_mode (bool): Whether to use binary mode (single output) or multi-class mode
            cache_images (bool): Whether to cache all images in memory
        """
        self.data_frame = dataframe
        self.img_dir = img_dir
        self.binary_mode = binary_mode
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.cache_images = cache_images
        self.cache_size = len(dataframe) + 100
    
        # Initialize LRU cache with OrderedDict instead of regular dict
        self.cache = collections.OrderedDict() if cache_images else {}
        
        if cache_images:
            print(f"Pre-loading all {len(dataframe)} images into memory...")
            for idx in tqdm(range(len(dataframe))):
                # Get the image path
                image_name = self.data_frame.iloc[idx]['image_name']
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_name = image_name + '.jpg'
                
                img_path = os.path.join(self.img_dir, image_name)
                
                # Try to load the image
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.cache[idx] = image
                except FileNotFoundError:
                    # Try alternative extensions
                    found = False
                    for ext in ['.jpg', '.jpeg', '.png']:
                        alt_path = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx]['image_name']}{ext}")
                        if os.path.exists(alt_path):
                            image = Image.open(alt_path).convert('RGB')
                            self.cache[idx] = image
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: Could not find image for {self.data_frame.iloc[idx]['image_name']}")
            
            print(f"Successfully loaded {len(self.cache)} images into memory")
            
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image from cache if available
        if self.cache_images and idx in self.cache:
            # Move this item to the end (most recently used)
            image = self.cache.pop(idx)
            self.cache[idx] = image
        else:
            # Load image from disk
            image_name = self.data_frame.iloc[idx]['image_name']
            
            # Check if extension is missing and add .jpg if needed
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_name = image_name + '.jpg'
            
            # Create full path
            img_path = os.path.join(self.img_dir, image_name)
            
            # Try to open the image
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Add to cache if caching is enabled
                if self.cache_images:
                    # If cache is full, remove the oldest item (first in OrderedDict)
                    if len(self.cache) >= self.cache_size:
                        self.cache.popitem(last=False)
                    # Add current item to the end (most recently used)
                    self.cache[idx] = image
                    
            except FileNotFoundError:
                # If image not found, try with other extensions
                found = False
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = os.path.join(self.img_dir, f"{self.data_frame.iloc[idx]['image_name']}{ext}")
                    if os.path.exists(alt_path):
                        image = Image.open(alt_path).convert('RGB')
                        if self.cache_images:
                            if len(self.cache) >= self.cache_size:
                                self.cache.popitem(last=False)
                            self.cache[idx] = image
                        found = True
                        break
                
                if not found:
                    raise FileNotFoundError(f"Could not find image for: {self.data_frame.iloc[idx]['image_name']}")
        
        # Get the target label (skin type, convert to 0-based index for classification)
        target = self.data_frame.iloc[idx]['predicted_skin_type'] - 1
        
        if self.binary_mode:
            # For BCEWithLogitsLoss - single output with value 0 or 1
            target = torch.tensor(target, dtype=torch.float).unsqueeze(0)
        else:
            # For CrossEntropyLoss - class index (no one-hot encoding needed)
            target = torch.tensor(target, dtype=torch.long)
        
        # Apply transforms (transforms are applied on-the-fly to allow for data augmentation)
        if self.transform:
            image = self.transform(image)
            
        return image, target

class SkinTypeCNN(nn.Module):
    """CNN model for skin type classification."""
    
    def __init__(self, num_classes=MONK_SKIN_TYPES, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of skin type classes
            pretrained: Whether to use pretrained weights
        """
        super(SkinTypeCNN, self).__init__()
        
        # Load EfficientNet B0 as the base model
        if pretrained:
            self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.base_model = efficientnet_b0(weights=None)
        
        # Replace the final fully connected layer with a new one
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.base_model(x)

def train_cnn_model(train_df, val_df, test_df, image_folder, output_folder, batch_size=32, num_epochs=20, 
                   learning_rate=0.001, num_classes=MONK_SKIN_TYPES, pretrained=True, gpu_id=0, cache_images=True):
    """
    Train a CNN model for skin type classification.
    
    Args:
        train_df: DataFrame with training data
        val_df: DataFrame with validation data
        test_df: DataFrame with test data
        image_folder: Folder containing the images
        output_folder: Folder to save the model and results
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        num_classes: Number of skin type classes
        pretrained: Whether to use pretrained weights
        gpu_id: ID of the GPU to use (0, 1, or 2)
        cache_images: Whether to cache images in memory
        
    Returns:
        Trained CNN model
    """
    print("Training CNN model...")
    
    # Set GPU device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    # Define transforms
    train_transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Creating datasets with {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
    
    # Create datasets using CachedSkinDataset
    train_dataset = CachedSkinDataset(
        dataframe=train_df,
        img_dir=image_folder,
        transform=train_transform,
        binary_mode=False,  # Use multi-class mode for skin type classification
        cache_images=cache_images
    )
    
    val_dataset = CachedSkinDataset(
        dataframe=val_df,
        img_dir=image_folder,
        transform=val_transform,
        binary_mode=False,
        cache_images=cache_images
    )
    
    test_dataset = CachedSkinDataset(
        dataframe=test_df,
        img_dir=image_folder,
        transform=val_transform,
        binary_mode=False,
        cache_images=cache_images
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=8, 
                              pin_memory=True, 
                              persistent_workers=True,
                              prefetch_factor=2)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=8, 
                            pin_memory=True, 
                            persistent_workers=True,
                            prefetch_factor=2)
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=8, 
                             pin_memory=True, 
                             persistent_workers=True,
                             prefetch_factor=2)
    
    # Initialize model
    model = SkinTypeCNN(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Create directories for outputs
    os.makedirs(output_folder, exist_ok=True)
    plots_dir = os.path.join(output_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (train)"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch statistics
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (val)"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch statistics
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_folder, 'cnn_model_best.pth'))
            print(f"Saved best model with validation loss {val_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_folder, 'cnn_model_final.pth'))
    
    # Plot and save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'cnn_training_loss.png'))
    plt.close()
    
    # Plot and save accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'cnn_training_accuracy.png'))
    plt.close()

    # Load the best model state for evaluation
    model.load_state_dict(torch.load(os.path.join(output_folder, 'cnn_model_best.pth'), weights_only=True))
    
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating on test set"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to actual skin type values (1-10)
    all_preds = [p + 1 for p in all_preds]
    all_targets = [t + 1 for t in all_targets]
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (CNN Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(plots_dir, 'cnn_confusion_matrix.png'))
    plt.close()
    
    # Generate and save classification report
    report = classification_report(all_targets, all_preds)
    with open(os.path.join(output_folder, 'cnn_classification_report.txt'), 'w') as f:
        f.write("CNN MODEL CLASSIFICATION REPORT\n")
        f.write("===============================\n\n")
        f.write(report)
    
    return model

def predict_with_cnn(image_path, model_path=None, device=None, gpu_id=0):
    """
    Predict skin type for a single image using the CNN model.
    
    Args:
        image_path: Path to the image
        model_path: Path to the trained model
        device: PyTorch device (cpu or cuda)
        gpu_id: ID of GPU to use
        
    Returns:
        Predicted skin type (1-10) and confidence
    """
    # Set GPU device if not provided
    if device is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
                gpu_id = 0
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available. Using CPU.")
    
    # Try to find the model if not provided
    if model_path is None:
        # Look in common locations
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'trained_model', 'cnn_model.pth')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("Error: No CNN model found. Please provide a model path.")
            return None, None
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Could not load image {image_path}: {e}")
        return None, None
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image = transform(image).unsqueeze(0).to(device)
    
    # Load model
    try:
        # First try to load just the state dict
        model = SkinTypeCNN(num_classes=MONK_SKIN_TYPES)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading state dict: {e}")
        try:
            # Try to load the full model
            model = torch.load(model_path, map_location=device)
        except Exception as e2:
            print(f"Error loading full model: {e2}")
            return None, None
    
    # Set model to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Convert to actual skin type value (1-10)
    skin_type = predicted.item() + 1
    confidence = confidence.item()
    
    return skin_type, confidence

def cnn_batch_predict(df, image_folder, output_folder, model_output_folder, model_path=None, gpu_id=0, batch_size=64):
    """
    Use a trained CNN model to predict skin types for all images in the DataFrame.
    
    Args:
        df: DataFrame with image names
        image_folder: Folder containing the images
        output_folder: Folder to save results
        model_path: Path to the trained model
        gpu_id: ID of GPU to use (0, 1, or 2)
        batch_size: Batch size for prediction
        
    Returns:
        DataFrame with predictions added
    """
    print(f"Predicting skin types for {len(df)} images using CNN model...")
    
    # Set device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus:
            print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    # Load model
    if model_path is None:
        # Try to find model in default locations
        possible_paths = [
            os.path.join(model_output_folder, 'cnn_model_best.pth'),
            os.path.join(model_output_folder, 'cnn_model_final.pth'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Found model at {model_path}")
                break
                
        if model_path is None:
            print("Error: No CNN model found. Please train a model first.")
            return df
    
    try:
        # Load model
        model = SkinTypeCNN(num_classes=MONK_SKIN_TYPES)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return df
    
    # Define transform for prediction
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader for prediction
    # We'll use -1 as a placeholder for the skin type since we don't know it yet
    pred_df = df.copy()
    if 'predicted_skin_type' not in pred_df.columns:
        pred_df['predicted_skin_type'] = -1
        
    pred_dataset = CachedSkinDataset(
        dataframe=pred_df,
        img_dir=image_folder,
        transform=transform,
        binary_mode=False,
        cache_images=True
    )
    
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Make predictions
    all_preds = []
    all_confidences = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(pred_loader, desc="Making predictions"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predicted class and confidence
            confs, preds = torch.max(probabilities, dim=1)
            
            # Convert to actual skin type values (1-10)
            all_preds.extend((preds + 1).cpu().numpy())
            all_confidences.extend(confs.cpu().numpy())
    
    # Add predictions to DataFrame
    pred_df['cnn_predicted_skin_type'] = all_preds
    pred_df['cnn_confidence'] = all_confidences
    
    # Save predictions
    pred_path = os.path.join(output_folder, 'cnn_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved CNN predictions to {pred_path}")
    
    return pred_df

if __name__ == "__main__":
    # This allows testing the CNN model independently
    parser = argparse.ArgumentParser(description="Train or predict with CNN model")
    parser.add_argument("--train", action="store_true", help="Train the CNN model")
    parser.add_argument("--predict", help="Predict using an image path")
    parser.add_argument("--model", help="Path to model for prediction")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0, 1, or 2)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no_cache", action="store_true", help="Disable image caching")
    
    args = parser.parse_args()
    
    if args.train:
        print("Please use the main script to train the CNN model.")
    elif args.predict:
        skin_type, confidence = predict_with_cnn(args.predict, args.model, gpu_id=args.gpu)
        if skin_type:
            print(f"Predicted skin type: {skin_type} (confidence: {confidence:.4f})")
        else:
            print("Prediction failed.")
    else:
        parser.print_help()