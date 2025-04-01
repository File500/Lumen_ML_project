import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import collections


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
    
        # Initialize LRU cache with OrderedDict instead of regular dict
        self.cache = collections.OrderedDict() if cache_images else {}
        
        # No pre-loading, images will be cached on-demand
        if cache_images:
            print(f"Using LRU cache with maximum size of {5000} images")
        
        # Initialize cache
        #self.cache = {}
        '''
        # Pre-load all images into memory if caching is enabled
        if cache_images:
            print(f"Caching {len(dataframe)} images in memory...")
            for idx in tqdm(range(len(dataframe))):
                img_name = dataframe.iloc[idx, 0]
                
                # Check if extension is missing and add .jpg if needed
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = img_name + '.jpg'
                
                # Create full path
                img_path = os.path.join(img_dir, img_name)
                
                # Load image
                try:
                    image = Image.open(img_path).convert('RGB')
                    # Store the untransformed image (transforms will be applied on-the-fly)
                    self.cache[idx] = image
                except FileNotFoundError:
                    print(f"Warning: Image not found at {img_path}")
            
            print(f"Successfully cached {len(self.cache)} images")
        '''            
            
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


def load_model(model_path, model, optimizer):
    model_dict = torch.load(model_path)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("cuda not available!")
        return None, None, None

    model.load_state_dict(model_dict['model_state'])
    model.to(device)
    model_epoch = model_dict['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer, model_epoch


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


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    return device


class ModelTrainer:
    def __init__(self, dataset_path, csv_file, img_dir, training_batch_size, num_workers, shuffle,
            train_val_test_split=[0.7, 0.15, 0.15], loss_fn=None, epochs_to_train=10, 
            model=None, optimizer=None, start_epoch=1, binary_mode=True):
    
        self.binary_mode = binary_mode

        # Load the CSV file
        full_df = pd.read_csv(csv_file)
        
        # Check class distribution in the dataset
        benign_count = sum(full_df['target'] == 0)
        malignant_count = sum(full_df['target'] == 1)
        print(f"Dataset class distribution:")
        print(f"  Benign: {benign_count} ({benign_count/len(full_df)*100:.2f}%)")
        print(f"  Malignant: {malignant_count} ({malignant_count/len(full_df)*100:.2f}%)")
        
        # Perform stratified train/val/test split to maintain class distribution
        train_size, val_size, test_size = train_val_test_split
        
        # First split into train and temp (val+test combined) with stratification
        train_df, temp_df = train_test_split(
            full_df, 
            test_size=(val_size + test_size),
            stratify=full_df['target'],  # This ensures the same ratio of malignant samples
            random_state=42
        )
        
        # Then split temp into val and test with stratification
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(test_size / (val_size + test_size)),
            stratify=temp_df['target'],  # Maintain class distribution in val and test sets
            random_state=42
        )
        
        # Verify class distribution in each split
        print(f"Training set: {len(train_df)} samples")
        print(f"  Benign: {sum(train_df['target'] == 0)} ({sum(train_df['target'] == 0)/len(train_df)*100:.2f}%)")
        print(f"  Malignant: {sum(train_df['target'] == 1)} ({sum(train_df['target'] == 1)/len(train_df)*100:.2f}%)")
        
        print(f"Validation set: {len(val_df)} samples")
        print(f"  Benign: {sum(val_df['target'] == 0)} ({sum(val_df['target'] == 0)/len(val_df)*100:.2f}%)")
        print(f"  Malignant: {sum(val_df['target'] == 1)} ({sum(val_df['target'] == 1)/len(val_df)*100:.2f}%)")
        
        print(f"Test set: {len(test_df)} samples") 
        print(f"  Benign: {sum(test_df['target'] == 0)} ({sum(test_df['target'] == 0)/len(test_df)*100:.2f}%)")
        print(f"  Malignant: {sum(test_df['target'] == 1)} ({sum(test_df['target'] == 1)/len(test_df)*100:.2f}%)")
        
        # Save split datasets for reference
        train_df.to_csv(os.path.join(dataset_path, 'train_split.csv'),index=False)
        val_df.to_csv(os.path.join(dataset_path, 'val_split.csv'), index=False)
        test_df.to_csv(os.path.join(dataset_path, 'test_split.csv'), index=False)
        
        print(f"Data split complete.")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        # Data augmentation for training
        train_transform = transforms.Compose([
            # not needed now (already using scaled Dset)
            # transforms.Resize((224, 224)),

            # Random transforms with individual probabilities
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.6),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.4),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.3),

            # Always convert to tensor and normalize at the end
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])

        # For validation and test (no augmentation)
        eval_transform = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

        # used for oversampling class 1 10% / 90% class 0
        sampler = create_balanced_sampler(train_dataset, target_class=1, target_ratio=0.1)
        
        # Create data loaders
        self.training_dataloader = create_data_loader(train_dataset, training_batch_size, num_workers, sampler)
        self.validation_dataloader = create_data_loader(val_dataset, training_batch_size, num_workers)
        self.test_dataloader = create_data_loader(test_dataset, training_batch_size, num_workers)

        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.epochs_to_train = epochs_to_train
        self.start_epoch = start_epoch
        self.device = get_device()
        self.test_df = test_df  # Save for later use

        if model is None:
            print("Warning: No model provided to ModelTrainer")
            self.model = None
        else:
            self.model = model.to(self.device)

        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model_and_optimizer(self, model, optimizer):
        self.set_model(model)
        self.set_optimizer(optimizer)

    def create_model_state_dict(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
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

    def train_epoch(self):
        self.model.train()
        train_loss = []
        all_preds = []
        all_labels = []

        loop = tqdm(self.training_dataloader, leave=False)
        for image_batch, labels in loop:
            image_batch = image_batch.to(self.device)
            labels = labels.to(self.device)

            predicted_data = self.model(image_batch)

            loss = self.loss_fn(predicted_data, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.item())
            train_loss.append(loss.detach().cpu().numpy())

            # Handle predictions based on model output
            if len(predicted_data.shape) == 1 or predicted_data.shape[1] == 1:  # Binary with single output
                preds = torch.sigmoid(predicted_data).round().detach().cpu().numpy()
            else:  # Multi-class with multiple outputs
                preds = torch.argmax(predicted_data, dim=1, keepdim=True).detach().cpu().numpy()
                
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

        # Flatten lists for metric calculation
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        accuracy = accuracy_score(all_labels, all_preds)
        # Explicitly specify positive class as 1 (malignant)
        precision = precision_score(all_labels, all_preds, zero_division=0, pos_label=1)
        recall = recall_score(all_labels, all_preds, zero_division=0, pos_label=1)
        f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

        return np.mean(train_loss), accuracy, precision, recall, f1

    def evaluate(self, dataloader, desc="Evaluating"):
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
                loss = self.loss_fn(predicted_data, labels)
                losses.append(loss.detach().cpu().numpy())

                # Handle predictions based on model output
                if len(predicted_data.shape) == 1 or predicted_data.shape[1] == 1:  # Binary with single output
                    preds = torch.sigmoid(predicted_data).round().cpu().numpy()
                else:  # Multi-class with multiple outputs
                    preds = torch.argmax(predicted_data, dim=1, keepdim=True).cpu().numpy()
                    
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                loop.set_postfix(loss=loss.item())

            # Flatten lists for metric calculation
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            
            accuracy = accuracy_score(all_labels, all_preds)
            # Explicitly specify positive class as 1 (malignant)
            precision = precision_score(all_labels, all_preds, zero_division=0, pos_label=1)
            recall = recall_score(all_labels, all_preds, zero_division=0, pos_label=1)
            f1 = f1_score(all_labels, all_preds, zero_division=0, pos_label=1)

        return np.mean(losses), accuracy, precision, recall, f1
    
    def validate_epoch(self):
        return self.evaluate(self.validation_dataloader, desc="Validating")

    def plot_roc_curve(self, dataloader=None, device=None):
        # Use test_dataloader if no dataloader provided
        if dataloader is None:
            dataloader = self.test_dataloader
        
        # Rest of the existing implementation remains the same
        # Replace `model` with `self.model`
        device = device or self.device
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation
        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to appropriate device
                images = images.to(device)
                labels = labels.to(device)
                
                # Get model predictions (probabilities)
                outputs = torch.sigmoid(self.model(images)).cpu().numpy()
                
                # Flatten labels
                labels = labels.cpu().numpy().flatten()
                
                all_preds.extend(outputs)
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
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Save the plot
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(current_dir)
        save_path = os.path.join(project_path, "trained_model", "melanoma_classifier")
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
        plt.close()
        
        return roc_auc
    
    def plot_confusion_matrix(self, dataloader=None, device=None, class_names=None):
        """
        Generate and plot confusion matrix for a PyTorch model
        
        Args:
        - dataloader: DataLoader with test/validation data. Defaults to test_dataloader
        - device: Torch device (cuda/cpu). If None, will auto-detect.
        - class_names: List of class names. Defaults to ['Benign', 'Malignant']
        
        Returns:
        - Confusion matrix as numpy array
        """
        # Use test_dataloader if no dataloader provided
        if dataloader is None:
            dataloader = self.test_dataloader
        
        # Use class device if not specified
        device = device or self.device
        
        # Default class names if not provided
        if class_names is None:
            class_names = ['Benign', 'Malignant']
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Collect all predictions and true labels
        all_preds = []
        all_labels = []
        
        # Disable gradient calculation
        with torch.no_grad():
            for images, labels in dataloader:
                # Move data to appropriate device
                images = images.to(device)
                labels = labels.to(device)
                
                # Get model predictions
                outputs = self.model(images)
                
                # For binary classification with sigmoid
                preds = torch.sigmoid(outputs).round().cpu().numpy()
                
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
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    cbar=True)
        
        plt.title('Confusion Matrix for Melanoma Classification', fontsize=15)
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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(current_dir)
        save_path = os.path.join(project_path, "trained_model", "melanoma_classifier")
        os.makedirs(save_path, exist_ok=True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print out confusion matrix details
        print("\nConfusion Matrix Details:")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")
        
        return cm
    def plot_training_history(self, history):
        """
        Plot training and validation metrics over epochs
        
        Args:
            history: Dictionary containing lists of metrics per epoch
        """
        # Create directory for saving plots
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(current_dir)
        save_path = os.path.join(project_path, "trained_model", "melanoma_classifier")
        os.makedirs(save_path, exist_ok=True)
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot Loss
        axs[0, 0].plot(history['train_loss'], label='Training Loss')
        axs[0, 0].plot(history['val_loss'], label='Validation Loss')
        axs[0, 0].set_title('Loss over Epochs', fontsize=15)
        axs[0, 0].set_xlabel('Epoch', fontsize=12)
        axs[0, 0].set_ylabel('Loss', fontsize=12)
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot Accuracy
        axs[0, 1].plot(history['train_acc'], label='Training Accuracy')
        axs[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axs[0, 1].set_title('Accuracy over Epochs', fontsize=15)
        axs[0, 1].set_xlabel('Epoch', fontsize=12)
        axs[0, 1].set_ylabel('Accuracy', fontsize=12)
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot Precision and Recall
        axs[1, 0].plot(history['train_prec'], label='Training Precision')
        axs[1, 0].plot(history['val_prec'], label='Validation Precision')
        axs[1, 0].plot(history['train_rec'], label='Training Recall')
        axs[1, 0].plot(history['val_rec'], label='Validation Recall')
        axs[1, 0].set_title('Precision and Recall over Epochs', fontsize=15)
        axs[1, 0].set_xlabel('Epoch', fontsize=12)
        axs[1, 0].set_ylabel('Score', fontsize=12)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        
        # Plot F1 Score
        axs[1, 1].plot(history['train_f1'], label='Training F1')
        axs[1, 1].plot(history['val_f1'], label='Validation F1')
        axs[1, 1].set_title('F1 Score over Epochs', fontsize=15)
        axs[1, 1].set_xlabel('Epoch', fontsize=12)
        axs[1, 1].set_ylabel('F1 Score', fontsize=12)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300)
        plt.close()
        
    def test_model(self):
        """Evaluate the model on the test set"""
        test_loss, test_acc, test_prec, test_rec, test_f1 = self.evaluate(
            self.test_dataloader, 
            desc="Testing"
        )

        # Calculate ROC AUC
        roc_auc = self.plot_roc_curve()

        confusion_mat = self.plot_confusion_matrix(
            class_names=['Benign', 'Malignant']
        )
        
        print("\n----- Test Results -----")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        # Convert NumPy float32 to Python float
        test_results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(test_f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': {
                'true_negatives': int(confusion_mat[0, 0]),
                'false_positives': int(confusion_mat[0, 1]),
                'false_negatives': int(confusion_mat[1, 0]),
                'true_positives': int(confusion_mat[1, 1])
            }
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(current_dir)
        save_path =  os.path.join(project_path, "trained_model", "melanoma_classifier")
        
        os.makedirs(save_path, exist_ok=True)

        # Save as JSON
        import json
        with open(os.path.join(save_path, "test_results.json"), 'w') as f:
            json.dump(test_results, f, indent=4)
            
        return test_results
    
    def train(self):
        if self.model is None or self.optimizer is None:
            print("Model and/or Optimizer not initialized")
            return
        
        print("Starting Training...")
        
        best_val_loss = float('inf')
        best_model_state = None
        last_model_state = None

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_prec': [], 'val_prec': [],
            'train_rec': [], 'val_rec': [],
            'train_f1': [], 'val_f1': []
        }
        
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs_to_train):
            print(f"\nEpoch {epoch}/{self.start_epoch + self.epochs_to_train - 1}")

            # Measure training time
            train_start = time.time()
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch()
            train_end = time.time()
            print(f"Training time: {train_end - train_start:.2f} seconds")
            
            # Measure validation time
            val_start = time.time()
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate_epoch()
            val_end = time.time()
            print(f"Validation time: {val_end - val_start:.2f} seconds") 

            # Save metrics to history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_prec'].append(train_prec)
            history['val_prec'].append(val_prec)
            history['train_rec'].append(train_rec)
            history['val_rec'].append(val_rec)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1) 

            metrics_start = time.time()
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
            metrics_end = time.time()
            print(f"metrics time: {metrics_end - metrics_start:.2f} seconds") 

            # Prepare metrics
            train_metrics = (train_acc, train_prec, train_rec, train_f1)
            val_metrics = (val_acc, val_prec, val_rec, val_f1)
            
            # Create model state dictionary for the current epoch
            model_state_start = time.time()
            current_model_state = self.create_model_state_dict(epoch, train_loss, val_loss, train_metrics, val_metrics)
            model_state_end = time.time()
            print(f"metrics time: {model_state_end - model_state_start:.2f} seconds") 
            
            # Always save the last model state
            last_model_state = current_model_state
            
            # Check if this is the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = current_model_state

        self.plot_training_history(history)

        # Save path for models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_path = os.path.dirname(os.path.dirname(current_dir))  
        save_path = os.path.join(project_path, "Lumen_ML_project", "trained_model", "melanoma_classifier")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save the last model
        if last_model_state:
            torch.save(last_model_state, os.path.join(save_path, "last-model.pth"))
            print(f"\nLast model saved to {os.path.join(save_path, 'last-model.pth')}")
        
        # Save the best model
        if best_model_state:
            torch.save(best_model_state, os.path.join(save_path, "best-model.pth"))
            print(f"Best model saved to {os.path.join(save_path, 'best-model.pth')}")


        print("Finished training")
        
        # Load the best model for evaluation
        best_model_path = os.path.join(save_path, "best-model.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for evaluation...")
            best_model_state = torch.load(best_model_path)
            self.model.load_state_dict(best_model_state['model_state'])
        else:
            print("Warning: Best model file not found. Evaluating the current model instead.")
        
        # After training is complete, evaluate on the test set
        print("\nEvaluating model on test set...")
        return self.test_model()