import torch
import numpy as np
import copy
import time
from tqdm import tqdm
import os
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from utils import EarlyStopping, plot_training_history

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device=None, output_dir=None):
        """
        Initialize the model trainer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            output_dir: Directory to save outputs
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Ensure output directory exists
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def train(self, train_loader, val_loader, num_epochs=25, patience=10):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train for
            patience: Patience for early stopping
            
        Returns:
            Trained model and training history
        """
        since = time.time()
        
        # Initialize Early Stopping
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True
        )

        # Track best model
        best_model_wts = copy.deepcopy(self.model.state_dict())
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
        
        dataloaders = {'train': train_loader, 'val': val_loader}
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                    dataloader = dataloaders['train']
                else:
                    self.model.eval()
                    dataloader = dataloaders['val']
                
                running_loss = 0.0
                all_preds = []
                all_labels = []
                
                # Iterate over batches
                for inputs, labels in tqdm(dataloader, desc=phase):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward + optimize only in training phase
                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
                            self.optimizer.step()
                    
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
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(epoch_loss)
                        else:
                            self.scheduler.step()

                    # Early stopping and model checkpointing
                    early_stopping(epoch_loss)
                        
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                        # Save the best model
                        if self.output_dir:
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': epoch_loss,
                                'balanced_acc': epoch_balanced_acc,
                                'f1': f1,
                            }, os.path.join(self.output_dir, 'best_model.pth'))

            # Check for early stopping
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            print()
        
        # Save the final model
        if self.output_dir:
            torch.save({
                'epoch': num_epochs-1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
                'balanced_acc': epoch_balanced_acc,
            }, os.path.join(self.output_dir, 'final_model.pth'))
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {best_loss:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        # Plot training progress
        if self.output_dir:
            plot_training_history(history, output_dir=self.output_dir)
        
        return self.model, history