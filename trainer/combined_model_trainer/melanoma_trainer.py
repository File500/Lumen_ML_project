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
from model.modified_melanoma_model import ModifiedMelanomaClassifier

class MelanomaModelTrainer:
    def __init__(self, dataset_path, csv_file, img_dir, batch_size, num_workers, 
                 train_val_test_split=[0.7, 0.15, 0.15], epochs=10, learning_rate=0.001):
        
        self.device = get_device(Config.GPU_ID)
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
        save_dir = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{Config.TEST_NUM}")
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
        save_path = os.path.join("trained_model", f"feature_extractor_melanoma_classifier_test{Config.TEST_NUM}")
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