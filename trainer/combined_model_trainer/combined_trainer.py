import os
import sys
import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc, 
    confusion_matrix
)

# Local imports
from utils import get_device, clear_gpu_cache
from dataset import CachedMelanomaDataset
from losses import FocalLoss
from early_stopping import EarlyStopping
from transform import create_data_loader, create_balanced_sampler, stratified_split, get_random_transform
from config import Config

# Model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.skin_tone_model import EfficientNetB3SkinToneClassifier
from model.modified_melanoma_model import ModifiedMelanomaClassifier
from model.combined_model import CombinedTransferModel

class CombinedModelTrainer:
    def __init__(self, dataset_path, csv_file, img_dir, batch_size, num_workers, 
                 skin_model_path, melanoma_model_path, train_val_test_split=[0.7, 0.15, 0.15], 
                 epochs=10, learning_rate=0.001):
        
        self.device = get_device(Config.GPU_ID)
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
        save_dir = os.path.join("trained_model", f"combined_model_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"combined_model_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"combined_model_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"combined_model_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"combined_model_test{Config.TEST_NUM}")
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
            save_path = os.path.join("trained_model", f"combined_model_test{Config.TEST_NUM}")
            os.makedirs(save_path, exist_ok=True)
            
            import json
            with open(os.path.join(save_path, "combined_test_results.json"), 'w') as f:
                json.dump(test_results, f, indent=4)
            
            return test_results
