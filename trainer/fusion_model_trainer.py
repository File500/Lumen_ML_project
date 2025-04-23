import os
import sys
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import collections
import random
import gc
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import EfficientNetB3SkinToneClassifier
from model.modified_melanoma_model import ModifiedMelanomaClassifier
from model.combined_model import CombinedTransferModel

TEST_NUMBER = 1

# Dataset class from your existing code
class CachedMelanomaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None, binary_mode=True, cache_images=True):
        """
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
        self.cache_size = 5000
        
        # Initialize LRU cache with OrderedDict
        self.cache = collections.OrderedDict() if cache_images else {}
        
        if cache_images:
            print(f"Using LRU cache with maximum size of {self.cache_size} images")
            
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
            img_name = self.data_frame.iloc[idx, 0]
            
            # Check if extension is missing and add .jpg if needed
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'
            
            # Create full path
            img_path = os.path.join(self.img_dir, img_name)
            
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
                raise FileNotFoundError(f"Could not find image: {img_path}")
        
        # Get the target label (0 for benign, 1 for malignant)
        target = self.data_frame.iloc[idx, 8]  # Assuming 'target' is at column 8
        transform_type = self.data_frame.iloc[idx, 8]
        
        if self.binary_mode:
            # For BCEWithLogitsLoss - single output with value 0 or 1
            target = torch.tensor(target, dtype=torch.float).unsqueeze(0)
        else:
            # For CrossEntropyLoss - class index (no one-hot encoding needed)
            target = torch.tensor(target, dtype=torch.long)
        
        # Apply transforms (transforms are applied on-the-fly to allow for data augmentation)
        if isinstance(self.transform, list):
            image = self.transform[transform_type](image)
        else:
            image = self.transform(image)
            
        return image, target
    
class EarlyStopping:
    """Early stops the training if monitored metric doesn't improve."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', 
                 trace_func=print, monitor='loss', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.monitor = monitor  # 'loss', 'f1', 'recall', etc.
        self.mode = mode        # 'min' for loss, 'max' for metrics like F1, recall
        
        # Initialize best_val based on mode
        self.best_val = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, value, model, save_path):
        """
        value: The metric to monitor (loss value or F1 score)
        """
        # Determine if improvement based on mode
        if self.mode == 'min':
            is_better = value < self.best_val - self.delta
        else:  # 'max' mode
            is_better = value > self.best_val + self.delta
        
        if self.best_score is None or is_better:
            # Better score found
            self.best_score = value  # Store the actual value
            self.best_val = value
            self.save_checkpoint(value, model, save_path)
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, value, model, save_path):
        """Save model when monitored metric improves."""
        metric_name = "F1 score" if self.monitor == 'f1' else "Recall" if self.monitor == 'recall' else "Validation loss"
        
        if self.mode == 'min':
            direction = "decreased"
        else:  # 'max' mode
            direction = "increased"
            
        if self.verbose:
            self.trace_func(f'{metric_name} {direction} ({self.best_val:.6f} --> {value:.6f}).  Saving model ...')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            self.monitor: value  # Save the metric value with the appropriate key
        }, save_path)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# Helper functions from your existing code
def get_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Explicitly select GPU 2
        device = torch.device("cuda:1")
        
        # Optional: Verify the selected GPU is within available device count
        if device.index < torch.cuda.device_count():
            print(f"Using GPU {device.index}: {torch.cuda.get_device_name(device.index)}")
            return device
    
    # Fallback to CPU if CUDA is not available or selected GPU is invalid
    print("Using CPU")
    return torch.device("cpu")

def clear_gpu_cache(device=None):
    """
    Clear GPU cache and perform garbage collection
    """
    # Perform garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Optional: Synchronize the selected device if specified
        if device is not None:
            try:
                with torch.cuda.device(device):
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Error synchronizing device: {e}")
 
def get_random_transform():
    possible_transforms = [
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
    ]
    return random.choice(possible_transforms)

def create_balanced_sampler(dataset, target_class, target_ratio):
    # Get all labels from the dataset
    if hasattr(dataset, 'target'):
        labels = dataset.target
    else:
        return None

    # Calculate class counts
    class_count = np.bincount(labels)
    total_samples = len(labels)

    # Calculate the number of samples needed for target class to achieve desired ratio
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
    dataloader = DataLoader(data, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            sampler=sampler,
                            pin_memory=True, 
                            persistent_workers=True,
                            prefetch_factor=2
                        )
    return dataloader

def load_model(model_path, model, optimizer=None):
    model_dict = torch.load(model_path)
    device = get_device()

    model.load_state_dict(model_dict['model_state'])
    model.to(device)
    model_epoch = model_dict['epoch']

    if optimizer is not None and 'optimizer_state' in model_dict:
        optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer, model_epoch

def stratified_split(full_df, train_size, val_size, test_size, random_state=42):
    """
    Perform stratified split based on both target and skin type
    
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


# Class for training the modified melanoma model
class MelanomaModelTrainer:
    def __init__(self, dataset_path, csv_file, img_dir, batch_size, num_workers, 
                 train_val_test_split=[0.7, 0.15, 0.15], epochs=10, learning_rate=0.001):
        
        self.device = get_device()
        self.binary_mode = True
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        
        # Load the CSV file
        full_df = pd.read_csv(csv_file)
        
        # Check class distribution in the dataset
        benign_count = sum(full_df['target'] == 0)
        malignant_count = sum(full_df['target'] == 1)
        print(f"Dataset class distribution:")
        print(f"  Benign: {benign_count} ({benign_count/len(full_df)*100:.2f}%)")
        print(f"  Malignant: {malignant_count} ({malignant_count/len(full_df)*100:.2f}%)")
        
        # Perform stratified train/val/test split
        train_size, val_size, test_size = train_val_test_split
        
        # First split into train and temp (val+test combined) with stratification
        train_df, val_df, test_df = stratified_split(
            full_df, 
            train_size=train_size, 
            val_size=val_size, 
            test_size=test_size, 
            random_state=42
        )
        
        # Save split datasets for reference
        os.makedirs(dataset_path, exist_ok=True)
        train_df.to_csv(os.path.join(dataset_path, 'train_split.csv'), index=False)
        val_df.to_csv(os.path.join(dataset_path, 'val_split.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_path, 'test_split.csv'), index=False)
        
        # Data transformations
        train_transform_minor = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.4),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.3),
            transforms.ToTensor()
        ])
    
        train_transform_major = transforms.Compose([
            get_random_transform(),
            transforms.ToTensor()
        ])
        
        train_transform = [train_transform_major, train_transform_minor]  # transforms for 0 and 1 class

        # For validation and test (no augmentation)
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Create datasets
        train_dataset = CachedMelanomaDataset(
            dataframe=train_df, 
            img_dir=img_dir, 
            transform=train_transform, 
            binary_mode=self.binary_mode,
            cache_images=True
        )

        val_dataset = CachedMelanomaDataset(
            dataframe=val_df, 
            img_dir=img_dir, 
            transform=eval_transform, 
            binary_mode=self.binary_mode,
            cache_images=True
        )

        test_dataset = CachedMelanomaDataset(
            dataframe=test_df, 
            img_dir=img_dir, 
            transform=eval_transform, 
            binary_mode=self.binary_mode,
            cache_images=True
        )

        # Create sampler for balancing classes
        sampler = create_balanced_sampler(train_dataset, target_class=1, target_ratio=0.3)
        
        # Create data loaders
        self.train_loader = create_data_loader(train_dataset, batch_size, num_workers, sampler)
        self.val_loader = create_data_loader(val_dataset, batch_size, num_workers)
        self.test_loader = create_data_loader(test_dataset, batch_size, num_workers)
        
        # Initialize the modified melanoma model
        self.model = ModifiedMelanomaClassifier(num_classes=2, binary_mode=True)
        self.model.to(self.device)
        
        # Define loss function
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = FocalLoss(gamma=2.0, alpha=0.75)
        #self.criterion = FocalLoss(gamma=4.0, alpha=0.85)
        
        # Initialize history for tracking metrics
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_f1': [], 'val_f1': []
        }
        
    def load_pretrained_weights(self, original_model_path):
        """
        Load weights from original melanoma model and transfer compatible weights
        """
        try:
            # Load the original state dict
            original_state_dict = torch.load(original_model_path)
            
            # If it's a full model state dict with 'model_state' key
            if 'model_state' in original_state_dict:
                original_state_dict = original_state_dict['model_state']
            
            # Create a new state dict for the modified model
            modified_state_dict = self.model.state_dict()
            
            # Transfer weights for all layers except the classifier
            pretrained_dict = {k: v for k, v in original_state_dict.items() 
                              if 'classifier' not in k and k in modified_state_dict}
            
            # Update the model with the pretrained weights
            modified_state_dict.update(pretrained_dict)
            self.model.load_state_dict(modified_state_dict, strict=False)
            
            print(f"Successfully loaded pretrained weights from {original_model_path}")
            print(f"Transferred {len(pretrained_dict)} layers, excluding classifier layers")
            
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    def train_model(self):
        """
        Train the modified melanoma model using a two-phase approach with early stopping
        """
        # Phase 1: Train only the classifier
        optimizer_phase1 = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.05)
        
        # Phase 2: Fine-tune the whole model
        optimizer_phase2 = optim.AdamW(self.model.parameters(), lr=self.learning_rate/10, weight_decay=0.05)
        
        # Learning rate scheduler for phase 2
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_phase2, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Create save directory if it doesn't exist
        save_dir = os.path.join("trained_model", f"melanoma_classifier_test{TEST_NUMBER}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize early stopping for Phase 1
        
        early_stopping_phase1 = EarlyStopping(
            patience=10,  # Increase from 10
            verbose=True,
            delta=0.001,
            path=os.path.join(save_dir, 'checkpoint_phase1.pth'),
            monitor='loss',  # Monitor f1 or  loss
            mode='min'     # Higher is better for F1
        )
        
        # Initialize early stopping for Phase 2
        early_stopping_phase2 = EarlyStopping(
            patience=10,  # Increase from 10
            verbose=True,
            delta=0.001,
            path=os.path.join(save_dir, 'checkpoint_phase2.pth'),
            monitor='f1',  # Monitor F1 or loss
            mode='max'     # Higher is better for F1
        )
        
        best_val_loss = float('inf')
        best_val_f1 = -float('inf')
        best_model_state = None
        
        # Determine the split between phases
        phase1_epochs = min(50, self.epochs)
        
        print("Starting Modified Melanoma Model Training...")
        
        # Phase 1: Train only the classifier
        print("\nPhase 1: Training only the classifier layers")
        
        # Freeze the backbone
        for param in self.model.efficientnet.features.parameters():
            param.requires_grad = True
        
        for epoch in range(phase1_epochs):
            print(f"\nPhase 1 - Epoch {epoch+1}/{phase1_epochs}")
            
            # Training
            self.model.train()
            train_loss, train_metrics = self._train_epoch(optimizer_phase1)
            
            # Validation
            val_loss, val_metrics = self._evaluate(self.val_loader)
            val_f1 = val_metrics[3]
            
            # Update history
            self._update_history(train_loss, val_loss, train_metrics, val_metrics)
            
            # Print metrics
            self._print_metrics(epoch+1, phase1_epochs, train_loss, val_loss, train_metrics, val_metrics, phase=1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self._create_model_state_dict(
                    epoch+1, train_loss, val_loss, train_metrics, val_metrics, optimizer_phase1
                )
            
            # Early stopping check
            early_stopping_phase1(val_loss, self.model, early_stopping_phase1.path)
            if early_stopping_phase1.early_stop:
                print("Early stopping triggered in Phase 1")
                break

            #if val_f1 > best_val_f1:
            #    best_val_f1 = val_f1
            #    best_model_state = self._create_model_state_dict(
            #        epoch+1, train_loss, val_loss, train_metrics, val_metrics
            #    )
            #    print(f"New best F1 score: {best_val_f1:.4f}")
        
            # Early stopping check using F1 score
            #early_stopping_phase1(val_f1, self.model, early_stopping_phase1.path)
            #if early_stopping_phase1.early_stop:
            #    print("Early stopping triggered in Phase 1")
            #    break

        # Clear GPU cache between phases
        clear_gpu_cache(self.device)

        # Phase 2: Fine-tune the whole model
        if self.epochs > phase1_epochs and not early_stopping_phase1.early_stop:
            print("\nPhase 2: Fine-tuning the entire model")
            
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
            
            for epoch in range(phase1_epochs, self.epochs):
                print(f"\nPhase 2 - Epoch {epoch+1}/{self.epochs}")
                
                # Training
                self.model.train()
                train_loss, train_metrics = self._train_epoch(optimizer_phase2)
                
                # Validation
                val_loss, val_metrics = self._evaluate(self.val_loader)
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Update history
                self._update_history(train_loss, val_loss, train_metrics, val_metrics)
                
                # Print metrics
                self._print_metrics(epoch+1, self.epochs, train_loss, val_loss, train_metrics, val_metrics, phase=2)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self._create_model_state_dict(
                        epoch+1, train_loss, val_loss, train_metrics, val_metrics, optimizer_phase2
                    )
                
                # Early stopping check
                early_stopping_phase2(val_loss, self.model, early_stopping_phase2.path)
                if early_stopping_phase2.early_stop:
                    print("Early stopping triggered in Phase 2")
                    break
        
        # Save the best model
        save_path = os.path.join("trained_model", f"melanoma_classifier_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        if best_model_state:
            torch.save(best_model_state, os.path.join(save_path, "modified_melanoma_model.pth"))
            print(f"\nBest model saved to {os.path.join(save_path, 'modified_melanoma_model.pth')}")
            
            # Load the best model for evaluation
            self.model.load_state_dict(best_model_state['model_state'])
        
        # Plot training history
        self._plot_training_history()
        
        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_results = self._test_model()

        clear_gpu_cache(self.device)
        
        return self.model, test_results
    
    def _train_epoch(self, optimizer):
        """Train for one epoch"""
        self.model.train()
        train_loss = []
        all_preds = []
        all_labels = []

        loop = tqdm(self.train_loader, leave=False)
        for image_batch, labels in loop:
            image_batch = image_batch.to(self.device)
            labels = labels.to(self.device)

            predicted_data = self.model(image_batch)

            loss = self.criterion(predicted_data, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.detach().cpu().numpy())

            # Get predictions
            preds = torch.sigmoid(predicted_data).round().detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        # Calculate metrics
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0, pos_label=1)
        recall = recall_score(all_labels, all_preds, zero_division=0, pos_label=1)
        f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

        return np.mean(train_loss), (accuracy, precision, recall, f1)
    
    def _evaluate(self, dataloader, desc="Evaluating"):
        """Evaluate the model on a dataloader"""
        self.model.eval()

        with torch.no_grad():
            losses = []
            all_preds = []
            all_labels = []

            loop = tqdm(dataloader, leave=False, desc=desc)
            for image_batch, labels in loop:
                image_batch = image_batch.to(self.device)
                labels = labels.to(self.device)

                predicted_data = self.model(image_batch)
                loss = self.criterion(predicted_data, labels)
                losses.append(loss.detach().cpu().numpy())

                # Get predictions
                preds = (torch.sigmoid(predicted_data) > 0.3).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                loop.set_postfix(loss=loss.item())

            # Calculate metrics
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0, pos_label=1)
            recall = recall_score(all_labels, all_preds, zero_division=0, pos_label=1)
            f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

        return np.mean(losses), (accuracy, precision, recall, f1)
    
    def _update_history(self, train_loss, val_loss, train_metrics, val_metrics):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_metrics[0])
        self.history['val_acc'].append(val_metrics[0])
        self.history['train_prec'].append(train_metrics[1])
        self.history['val_prec'].append(val_metrics[1])
        self.history['train_rec'].append(train_metrics[2])
        self.history['val_rec'].append(val_metrics[2])
        self.history['train_f1'].append(train_metrics[3])
        self.history['val_f1'].append(val_metrics[3])
    
    def _print_metrics(self, epoch, total_epochs, train_loss, val_loss, train_metrics, val_metrics, phase=1):
        """Print training and validation metrics"""
        print(f"Phase {phase} - Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics[0]:.4f}, Prec: {train_metrics[1]:.4f}, Rec: {train_metrics[2]:.4f}, F1: {train_metrics[3]:.4f}")
        print(f"Val Acc: {val_metrics[0]:.4f}, Prec: {val_metrics[1]:.4f}, Rec: {val_metrics[2]:.4f}, F1: {val_metrics[3]:.4f}")
    
    def _create_model_state_dict(self, epoch, train_loss, val_loss, train_metrics, val_metrics, optimizer):
        """Create a state dictionary for saving the model"""
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_name': 'AdamW',
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics[0],
            'train_precision': train_metrics[1],
            'train_recall': train_metrics[2],
            'train_f1': train_metrics[3],
            'val_accuracy': val_metrics[0],
            'val_precision': val_metrics[1],
            'val_recall': val_metrics[2],
            'val_f1': val_metrics[3],
        }
        return model_state
    
    def _plot_training_history(self):
        """Plot training and validation metrics over epochs"""
        # Create directory for saving plots
        save_path = os.path.join("trained_model", f"melanoma_classifier_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot Loss
        axs[0, 0].plot(self.history['train_loss'], label='Training Loss')
        axs[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss over Epochs', fontsize=15)
        axs[0, 0].set_xlabel('Epoch', fontsize=12)
        axs[0, 0].set_ylabel('Loss', fontsize=12)
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot Accuracy
        axs[0, 1].plot(self.history['train_acc'], label='Training Accuracy')
        axs[0, 1].plot(self.history['val_acc'], label='Validation Accuracy')
        axs[0, 1].set_title('Accuracy over Epochs', fontsize=15)
        axs[0, 1].set_xlabel('Epoch', fontsize=12)
        axs[0, 1].set_ylabel('Accuracy', fontsize=12)
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot Precision and Recall
        axs[1, 0].plot(self.history['train_prec'], label='Training Precision')
        axs[1, 0].plot(self.history['val_prec'], label='Validation Precision')
        axs[1, 0].plot(self.history['train_rec'], label='Training Recall')
        axs[1, 0].plot(self.history['val_rec'], label='Validation Recall')
        axs[1, 0].set_title('Precision and Recall over Epochs', fontsize=15)
        axs[1, 0].set_xlabel('Epoch', fontsize=12)
        axs[1, 0].set_ylabel('Score', fontsize=12)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot F1 Score
        axs[1, 1].plot(self.history['train_f1'], label='Training F1')
        axs[1, 1].plot(self.history['val_f1'], label='Validation F1')
        axs[1, 1].set_title('F1 Score over Epochs', fontsize=15)
        axs[1, 1].set_xlabel('Epoch', fontsize=12)
        axs[1, 1].set_ylabel('F1 Score', fontsize=12)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'melanoma_training_history.png'), dpi=300)
        plt.close()

    def _test_model(self):
        """Evaluate the model on the test set"""
        print("Testing melanoma model on test dataset...")
        
        # Evaluate on test set
        test_loss, test_metrics = self._evaluate(self.test_loader, desc="Testing")
        
        # Calculate ROC AUC and plot ROC curve
        roc_auc = self._plot_roc_curve()
        
        # Generate confusion matrix
        confusion_mat = self._plot_confusion_matrix()
        
        # Print test results
        print("\n----- Melanoma Model Test Results -----")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_metrics[0]:.4f}")
        print(f"Test Precision: {test_metrics[1]:.4f}")
        print(f"Test Recall: {test_metrics[2]:.4f}")
        print(f"Test F1 Score: {test_metrics[3]:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Prepare test results dictionary
        tn, fp, fn, tp = confusion_mat.ravel()
        test_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_metrics[0]),
            'test_precision': float(test_metrics[1]),
            'test_recall': float(test_metrics[2]),
            'test_f1': float(test_metrics[3]),
            'roc_auc': float(roc_auc),
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }
        
        # Save results to JSON
        save_path = os.path.join("trained_model", f"melanoma_classifier_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        import json
        with open(os.path.join(save_path, "melanoma_test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        return test_results

    def _plot_roc_curve(self, dataloader=None):
        """Generate and plot ROC curve"""
        # Use test_dataloader if no dataloader provided
        if dataloader is None:
            dataloader = self.test_loader
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation
        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to appropriate device
                images = images.to(self.device)
                
                # Get model predictions (probabilities)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # Flatten labels
                labels = labels.cpu().numpy().flatten()
                
                all_preds.extend(probs)
                all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Melanoma Classifier', fontsize=15)
        plt.legend(loc="lower right")
        
        # Save the plot
        save_path = os.path.join("trained_model", f"melanoma_classifier_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'melanoma_roc_curve.png'), dpi=300)
        plt.close()
        
        return roc_auc

    def _plot_confusion_matrix(self, dataloader=None):
        """Generate and plot confusion matrix"""
        # Use test_dataloader if no dataloader provided
        if dataloader is None:
            dataloader = self.test_loader
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation
        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to appropriate device
                images = images.to(self.device)
                
                # Get model predictions
                outputs = self.model(images)
                preds = (torch.sigmoid(outputs) > 0.3).float().cpu().numpy()
                
                # Flatten labels
                labels = labels.cpu().numpy().flatten()
                preds = preds.flatten()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Create a more detailed confusion matrix plot
        plt.figure(figsize=(10, 7))
        
        # Create heatmap with more detailed formatting
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malignant'], 
                    yticklabels=['Benign', 'Malignant'],
                    cbar=True)
        
        plt.title('Confusion Matrix - Melanoma Classifier', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add some explanatory text
        tn, fp, fn, tp = cm.ravel()
        plt.text(1.1, -0.1, 
                f'True Negatives: {tn}\n'
                f'False Positives: {fp}\n'
                f'False Negatives: {fn}\n'
                f'True Positives: {tp}', 
                horizontalalignment='left',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        
        # Save the plot
        save_path = os.path.join("trained_model", f"melanoma_classifier_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'melanoma_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print out confusion matrix details
        print("\nConfusion Matrix Details:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        return cm
    
# Class for training the combined transfer learning model
class CombinedModelTrainer:
    def __init__(self, dataset_path, csv_file, img_dir, batch_size, num_workers, 
                 skin_model_path, melanoma_model_path, train_val_test_split=[0.7, 0.15, 0.15], 
                 epochs=10, learning_rate=0.001):
        
        self.device = get_device()
        self.binary_mode = True
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Load the pre-trained models
        print("Loading pre-trained models...")
        
        # 1. Load skin tone classifier
        try:
            print(f"Loading skin tone classifier from {skin_model_path}")
            
            # Load the checkpoint
            skin_state_dict = torch.load(skin_model_path, map_location=self.device)
            
            # Initialize the model
            self.skin_model = EfficientNetB3SkinToneClassifier(num_classes=7)
            
            # Check and load state dict
            if isinstance(skin_state_dict, dict):
                if 'model_state' in skin_state_dict:
                    # Your original format
                    state_dict = skin_state_dict['model_state']
                elif 'state_dict' in skin_state_dict:
                    # Alternative format
                    state_dict = skin_state_dict['state_dict']
                elif 'model_state_dict' in skin_state_dict:
                    # Another common format
                    state_dict = skin_state_dict['model_state_dict']
                else:
                    # If the dict itself is the state dict
                    state_dict = skin_state_dict
            else:
                # If it's not a dictionary, assume it's the state dict
                state_dict = skin_state_dict
            
            # Load the state dictionary
            self.skin_model.load_state_dict(state_dict)
            
            # Move to device and set to evaluation mode
            self.skin_model = self.skin_model.to(self.device)
            self.skin_model.eval()
            
            print("Skin tone classifier loaded successfully")

        except Exception as e:
            print(f"Error loading skin tone classifier: {e}")
            # Fallback: create a new model
            self.skin_model = EfficientNetB3SkinToneClassifier(num_classes=7)
            self.skin_model = self.skin_model.to(self.device)
            self.skin_model.eval()
            print("Created a new skin tone classifier model")
        
        # 2. Load modified melanoma classifier
        self.melanoma_model = ModifiedMelanomaClassifier(num_classes=2, binary_mode=True)
        melanoma_state_dict = torch.load(melanoma_model_path)
        if 'model_state' in melanoma_state_dict:
            self.melanoma_model.load_state_dict(melanoma_state_dict['model_state'])
        else:
            self.melanoma_model.load_state_dict(melanoma_state_dict)
        self.melanoma_model.to(self.device)
        self.melanoma_model.eval()  # Set to evaluation mode
        
        # 3. Create the combined model
        self.combined_model = CombinedTransferModel(
            skin_tone_model=self.skin_model,
            melanoma_model=self.melanoma_model,
            num_classes=2,
            binary_mode=True
        )
        self.combined_model.to(self.device)
        
        # Load the CSV file
        full_df = pd.read_csv(csv_file)
        
        # Check class distribution in the dataset
        benign_count = sum(full_df['target'] == 0)
        malignant_count = sum(full_df['target'] == 1)
        print(f"Dataset class distribution:")
        print(f"  Benign: {benign_count} ({benign_count/len(full_df)*100:.2f}%)")
        print(f"  Malignant: {malignant_count} ({malignant_count/len(full_df)*100:.2f}%)")
        
        # Perform stratified train/val/test split
        train_size, val_size, test_size = train_val_test_split
        
        # First split into train and temp (val+test combined) with stratification
        train_df, val_df, test_df = stratified_split(
            full_df, 
            train_size=train_size, 
            val_size=val_size, 
            test_size=test_size, 
            random_state=42
        )
        
        # Save split datasets for reference
        os.makedirs(dataset_path, exist_ok=True)
        train_df.to_csv(os.path.join(dataset_path, 'combined_train_split.csv'), index=False)
        val_df.to_csv(os.path.join(dataset_path, 'combined_val_split.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_path, 'combined_test_split.csv'), index=False)
        
        # Data transformations
        train_transform_minor = transforms.Compose([
            transforms.RandomRotation(20),  # Increased rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))], p=0.4),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
            transforms.ToTensor()
        ])
    
        train_transform_major = transforms.Compose([
            get_random_transform(),
            transforms.ToTensor()
        ])
        
        train_transform = [train_transform_major, train_transform_minor]  # transforms for 0 and 1 class

        # For validation and test (no augmentation)
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Create datasets
        train_dataset = CachedMelanomaDataset(
            dataframe=train_df, 
            img_dir=img_dir, 
            transform=train_transform, 
            binary_mode=self.binary_mode,
            cache_images=True
        )

        val_dataset = CachedMelanomaDataset(
            dataframe=val_df, 
            img_dir=img_dir, 
            transform=eval_transform, 
            binary_mode=self.binary_mode,
            cache_images=True
        )

        test_dataset = CachedMelanomaDataset(
            dataframe=test_df, 
            img_dir=img_dir, 
            transform=eval_transform, 
            binary_mode=self.binary_mode,
            cache_images=True
        )

        # Create sampler for balancing classes
        sampler = create_balanced_sampler(train_dataset, target_class=1, target_ratio=0.3)
        
        # Create data loaders
        self.train_loader = create_data_loader(train_dataset, batch_size, num_workers, sampler)
        self.val_loader = create_data_loader(val_dataset, batch_size, num_workers)
        self.test_loader = create_data_loader(test_dataset, batch_size, num_workers)
        
        # Define loss function
        #self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = FocalLoss(gamma=2.0, alpha=0.75)
        #self.criterion = FocalLoss(gamma=4.0, alpha=0.85)

        # Initialize optimizer (only for the combined classifier)
        self.optimizer = optim.AdamW(self.combined_model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Initialize scheduler
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        #)

        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=learning_rate/10,  # 0.00003
            max_lr=learning_rate*5,    # 0.0015
            step_size_up=len(self.train_loader)*3,  # 3 epochs
            mode='triangular2'  # Decreases max_lr by half each cycle
        )
        
        # Initialize history for tracking metrics
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_f1': [], 'val_f1': []
        }

    def train_model(self):
        """Train the combined transfer learning model"""
        print("Starting Combined Transfer Learning Model Training...")
        
        best_val_loss = float('inf')
        best_val_f1 = -float('inf')
        best_model_state = None
        
        # Create save directory if it doesn't exist
        save_dir = os.path.join("trained_model", f"combined_model_test{TEST_NUMBER}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=10,  # Increase from 10
            verbose=True,
            delta=0.001,
            path=os.path.join(save_dir, 'checkpoint.pth'),
            monitor='f1',  # Monitor F1 instead of loss
            mode='max'     # Higher is better for F1
        )
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training
            train_loss, train_metrics = self._train_epoch()
            
            # Validation
            val_loss, val_metrics = self._evaluate(self.val_loader)

            val_f1 = val_metrics[3]
            
            # Update scheduler
            self.scheduler.step()
            
            # Update history
            self._update_history(train_loss, val_loss, train_metrics, val_metrics)
            
            # Print metrics
            self._print_metrics(epoch+1, self.epochs, train_loss, val_loss, train_metrics, val_metrics)
            
            # Save best model
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    best_model_state = self._create_model_state_dict(
            #        epoch+1, train_loss, val_loss, train_metrics, val_metrics
            #    )
            
            # Early stopping check
            #early_stopping(val_loss, self.combined_model, early_stopping.path)
            #if early_stopping.early_stop:
            #    print("Early stopping triggered!")
            #    break

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = self._create_model_state_dict(
                    epoch+1, train_loss, val_loss, train_metrics, val_metrics, self.optimizer
                )
                print(f"New best F1 score: {best_val_f1:.4f}")

            early_stopping(val_f1, self.combined_model, early_stopping.path)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        # Save the best model
        save_path = os.path.join("trained_model", f"combined_model_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        if best_model_state:
            torch.save(best_model_state, os.path.join(save_path, "best_combined_model.pth"))
            print(f"\nBest combined model saved to {os.path.join(save_path, 'best_combined_model.pth')}")
            
            # Load the best model for evaluation
            self.combined_model.load_state_dict(best_model_state['model_state'])
        
        # Plot training history
        self._plot_training_history()
        
        # Evaluate on test set
        print("\nEvaluating combined model on test set...")
        test_results = self._test_model()
        
        return self.combined_model, test_results
            
    def _train_epoch(self):
        """Train for one epoch"""
        self.combined_model.train()
        train_loss = []
        all_preds = []
        all_labels = []

        loop = tqdm(self.train_loader, leave=False)
        for image_batch, labels in loop:
            image_batch = image_batch.to(self.device)
            labels = labels.to(self.device)

            # Forward pass through the combined model
            outputs = self.combined_model(image_batch)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss.append(loss.detach().cpu().numpy())

            # Get predictions
            preds = torch.sigmoid(outputs).round().detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
            
            loop.set_postfix(loss=loss.item())

        # Calculate metrics
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0, pos_label=1)
        recall = recall_score(all_labels, all_preds, zero_division=0, pos_label=1)
        f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

        return np.mean(train_loss), (accuracy, precision, recall, f1)

    def _evaluate(self, dataloader, desc="Evaluating"):
        """Evaluate the model on a dataloader"""
        self.combined_model.eval()

        with torch.no_grad():
            losses = []
            all_preds = []
            all_labels = []

            loop = tqdm(dataloader, leave=False, desc=desc)
            for image_batch, labels in loop:
                image_batch = image_batch.to(self.device)
                labels = labels.to(self.device)

                outputs = self.combined_model(image_batch)
                loss = self.criterion(outputs, labels)
                losses.append(loss.detach().cpu().numpy())

                # Get predictions
                preds = (torch.sigmoid(outputs) > 0.3).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                loop.set_postfix(loss=loss.item())

            # Calculate metrics
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0, pos_label=1)
            recall = recall_score(all_labels, all_preds, zero_division=0, pos_label=1)
            f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

        return np.mean(losses), (accuracy, precision, recall, f1)

    def _update_history(self, train_loss, val_loss, train_metrics, val_metrics):
        """Update training history"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_metrics[0])
        self.history['val_acc'].append(val_metrics[0])
        self.history['train_prec'].append(train_metrics[1])
        self.history['val_prec'].append(val_metrics[1])
        self.history['train_rec'].append(train_metrics[2])
        self.history['val_rec'].append(val_metrics[2])
        self.history['train_f1'].append(train_metrics[3])
        self.history['val_f1'].append(val_metrics[3])

    def _print_metrics(self, epoch, total_epochs, train_loss, val_loss, train_metrics, val_metrics):
        """Print training and validation metrics"""
        print(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics[0]:.4f}, Prec: {train_metrics[1]:.4f}, Rec: {train_metrics[2]:.4f}, F1: {train_metrics[3]:.4f}")
        print(f"Val Acc: {val_metrics[0]:.4f}, Prec: {val_metrics[1]:.4f}, Rec: {val_metrics[2]:.4f}, F1: {val_metrics[3]:.4f}")

    def _create_model_state_dict(self, epoch, train_loss, val_loss, train_metrics, val_metrics, optimizer):
        """Create a state dictionary for saving the model"""
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.combined_model.state_dict(),
            'model_name': type(self.combined_model).__name__,
            'optimizer_name': 'AdamW',
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics[0],
            'train_precision': train_metrics[1],
            'train_recall': train_metrics[2],
            'train_f1': train_metrics[3],
            'val_accuracy': val_metrics[0],
            'val_precision': val_metrics[1],
            'val_recall': val_metrics[2],
            'val_f1': val_metrics[3],
        }
        return model_state
        
    def _plot_training_history(self):
        """Plot training and validation metrics over epochs"""
        # Create directory for saving plots
        save_path = os.path.join("trained_model", f"combined_model_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot Loss
        axs[0, 0].plot(self.history['train_loss'], label='Training Loss')
        axs[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss over Epochs', fontsize=15)
        axs[0, 0].set_xlabel('Epoch', fontsize=12)
        axs[0, 0].set_ylabel('Loss', fontsize=12)
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot Accuracy
        axs[0, 1].plot(self.history['train_acc'], label='Training Accuracy')
        axs[0, 1].plot(self.history['val_acc'], label='Validation Accuracy')
        axs[0, 1].set_title('Accuracy over Epochs', fontsize=15)
        axs[0, 1].set_xlabel('Epoch', fontsize=12)
        axs[0, 1].set_ylabel('Accuracy', fontsize=12)
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot Precision and Recall
        axs[1, 0].plot(self.history['train_prec'], label='Training Precision')
        axs[1, 0].plot(self.history['val_prec'], label='Validation Precision')
        axs[1, 0].plot(self.history['train_rec'], label='Training Recall')
        axs[1, 0].plot(self.history['val_rec'], label='Validation Recall')
        axs[1, 0].set_title('Precision and Recall over Epochs', fontsize=15)
        axs[1, 0].set_xlabel('Epoch', fontsize=12)
        axs[1, 0].set_ylabel('Score', fontsize=12)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot F1 Score
        axs[1, 1].plot(self.history['train_f1'], label='Training F1')
        axs[1, 1].plot(self.history['val_f1'], label='Validation F1')
        axs[1, 1].set_title('F1 Score over Epochs', fontsize=15)
        axs[1, 1].set_xlabel('Epoch', fontsize=12)
        axs[1, 1].set_ylabel('F1 Score', fontsize=12)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_training_history.png'), dpi=300)
        plt.close()

    def _plot_roc_curve(self, dataloader=None):
        """Generate and plot ROC curve"""
        # Use test_dataloader if no dataloader provided
        if dataloader is None:
            dataloader = self.test_loader
        
        # Set model to evaluation mode
        self.combined_model.eval()
        
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation
        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to appropriate device
                images = images.to(self.device)
                
                # Get model predictions (probabilities)
                outputs = self.combined_model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                # Flatten labels
                labels = labels.cpu().numpy().flatten()
                
                all_preds.extend(probs)
                all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Combined Transfer Learning Model')
        plt.legend(loc="lower right")
        
        # Save the plot
        save_path = os.path.join("trained_model", f"combined_model_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'combined_roc_curve.png'))
        plt.close()
        
        return roc_auc

    def _plot_confusion_matrix(self, dataloader=None):
        """Generate and plot confusion matrix"""
        # Use test_dataloader if no dataloader provided
        if dataloader is None:
            dataloader = self.test_loader
        
        # Set model to evaluation mode
        self.combined_model.eval()
        
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation
        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to appropriate device
                images = images.to(self.device)
                
                # Get model predictions
                outputs = self.combined_model(images)
                preds = (torch.sigmoid(outputs) > 0.3).float().cpu().numpy()
                
                # Flatten labels
                labels = labels.cpu().numpy().flatten()
                preds = preds.flatten()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Create a more detailed confusion matrix plot
        plt.figure(figsize=(10, 7))
        
        # Create heatmap with more detailed formatting
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Malignant'], 
                    yticklabels=['Benign', 'Malignant'],
                    cbar=True)
        
        plt.title('Confusion Matrix - Combined Transfer Learning Model', fontsize=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add some explanatory text
        tn, fp, fn, tp = cm.ravel()
        plt.text(1.1, -0.1, 
                f'True Negatives: {tn}\n'
                f'False Positives: {fp}\n'
                f'False Negatives: {fn}\n'
                f'True Positives: {tp}', 
                horizontalalignment='left',
                verticalalignment='center',
                transform=plt.gca().transAxes)
        
        # Save the plot
        save_path = os.path.join("trained_model", f"combined_model_test{TEST_NUMBER}")
        os.makedirs(save_path, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'combined_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print out confusion matrix details
        print("\nConfusion Matrix Details:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        return cm

    def _test_model(self):
            """Evaluate the model on the test set"""
            print("Testing combined model on test dataset...")
            
            # Evaluate on test set
            test_loss, test_metrics = self._evaluate(self.test_loader, desc="Testing")
            
            # Calculate ROC AUC
            roc_auc = self._plot_roc_curve()
            
            # Generate confusion matrix
            confusion_mat = self._plot_confusion_matrix()
            
            # Print test results
            print("\n----- Combined Model Test Results -----")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_metrics[0]:.4f}")
            print(f"Test Precision: {test_metrics[1]:.4f}")
            print(f"Test Recall: {test_metrics[2]:.4f}")
            print(f"Test F1 Score: {test_metrics[3]:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            
            # Prepare test results dictionary
            tn, fp, fn, tp = confusion_mat.ravel()
            test_results = {
                'test_loss': float(test_loss),
                'test_accuracy': float(test_metrics[0]),
                'test_precision': float(test_metrics[1]),
                'test_recall': float(test_metrics[2]),
                'test_f1': float(test_metrics[3]),
                'roc_auc': float(roc_auc),
                'confusion_matrix': {
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp)
                }
            }
            
            # Save results to JSON
            save_path = os.path.join("trained_model", f"combined_model_test{TEST_NUMBER}")
            os.makedirs(save_path, exist_ok=True)
            
            import json
            with open(os.path.join(save_path, "combined_test_results.json"), 'w') as f:
                json.dump(test_results, f, indent=4)
            
            return test_results



# Main function to run the complete pipeline
def run_transfer_learning_pipeline(config):
    """
    Run the complete transfer learning pipeline
    
    Args:
        config: Dictionary containing configuration parameters
    """
    # Create necessary directories
    os.makedirs(config["dataset_path"], exist_ok=True)
    os.makedirs(os.path.join("trained_model", "melanoma_classifier"), exist_ok=True)
    os.makedirs(os.path.join("trained_model", "combined_model"), exist_ok=True)
    
    # Step 1: Train the modified melanoma model with 128-neuron layer
    if config.get("train_melanoma", True):
        print("\n========== PHASE 1: TRAINING MODIFIED MELANOMA MODEL ==========\n")

        clear_gpu_cache()
        
        # Create the melanoma model trainer
        melanoma_trainer = MelanomaModelTrainer(
            dataset_path=config["dataset_path"],
            csv_file=config["csv_file"],
            img_dir=config["img_dir"],
            batch_size=config["melanoma_batch_size"],
            num_workers=config["num_workers"],
            epochs=config["melanoma_epochs"],
            learning_rate=config["melanoma_lr"]
        )
        
        # If pretrained original melanoma model exists, load its weights
        if "original_melanoma_model" in config and os.path.exists(config["original_melanoma_model"]):
            print(f"Loading weights from original melanoma model: {config['original_melanoma_model']}")
            melanoma_trainer.load_pretrained_weights(config["original_melanoma_model"])
        
        # Train the modified melanoma model
        trained_melanoma_model, melanoma_results = melanoma_trainer.train_model()
        
        # Set the path to the trained model for the next step
        melanoma_model_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{TEST_NUMBER}", "modified_melanoma_model.pth")

        clear_gpu_cache()

    else:
        # Use existing modified melanoma model
        melanoma_model_path = config["melanoma_model_path"]
        print(f"Skipping melanoma model training, using existing model: {melanoma_model_path}")
        # Initialize with None since we don't have results if skipping 
        melanoma_results = None
    
    # Step 2: Train the combined transfer learning model
    if config.get("train_combined", True):
        print("\n========== PHASE 2: TRAINING COMBINED TRANSFER LEARNING MODEL ==========\n")

        clear_gpu_cache()
        
        # Create the combined model trainer
        combined_trainer = CombinedModelTrainer(
            dataset_path=config["dataset_path"],
            csv_file=config["csv_file"],
            img_dir=config["img_dir"],
            batch_size=config["combined_batch_size"],
            num_workers=config["num_workers"],
            skin_model_path=config["skin_model_path"],
            melanoma_model_path=melanoma_model_path,
            epochs=config["combined_epochs"],
            learning_rate=config["combined_lr"]
        )
        
        # Train the combined model
        trained_combined_model, combined_results = combined_trainer.train_model()
        
        # Print final results
        print("\n========== FINAL RESULTS ==========\n")
        
        if melanoma_results:
            print("Modified Melanoma Model Results:")
            for key, value in melanoma_results.items():
                if key != 'confusion_matrix':
                    print(f"  {key}: {value:.4f}")
        
        print("\nCombined Transfer Learning Model Results:")
        for key, value in combined_results.items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value:.4f}")
        
        if melanoma_results:
            print("\nPerformance Improvement:")
            for key in melanoma_results:
                if key not in ['confusion_matrix', 'test_loss']:
                    diff = combined_results[key] - melanoma_results[key]
                    print(f"  {key}: {diff:.4f} ({'+' if diff > 0 else ''}{diff/melanoma_results[key]*100:.2f}%)")

        clear_gpu_cache()
    
    else:
        # Skip combined model training
        print(f"Skipping combined model training")
    
    print("\n========== TRANSFER LEARNING PIPELINE COMPLETE ==========\n")


# Example usage of the pipeline with a configuration dictionary
if __name__ == "__main__":
    # Configuration for the pipeline
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    PROJECT_PATH = os.path.dirname(CURRENT_DIR)

    DATA_PATH = os.path.join(PROJECT_PATH, "data")

    config = {
        # Dataset paths
        "dataset_path":  os.path.join(DATA_PATH, "dataset_splits"),
        "csv_file": os.path.join(DATA_PATH, "deduplicated_monk_scale_dataset_predictions.csv"),
        "img_dir":  os.path.join(DATA_PATH, "train_300X300_processed") ,
        
        # Model paths
        "skin_model_path": os.path.join(PROJECT_PATH, "trained_model", "skin_type_classifier", "EFNet_b3_300X300_final", "final_model.pth"),
        "melanoma_model_path": os.path.join(PROJECT_PATH, "trained_model", "feature_extractor_melanoma_classifier_final", "modified_melanoma_model.pth"), 
        
        # Training options
        "train_melanoma": True,
        "train_combined": True,
        
        # Training parameters
        "melanoma_batch_size": 30,
        "combined_batch_size": 16,
        "num_workers": 4,
        "melanoma_epochs": 50,
        "combined_epochs": 50,
        "melanoma_lr": 0.0003,
        "combined_lr": 0.0003,
    }
    
    # Uncomment to run the pipeline
    run_transfer_learning_pipeline(config)