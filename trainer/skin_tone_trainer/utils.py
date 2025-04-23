import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import time
from tqdm import tqdm

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

def save_metrics_to_file(metrics_dict, model_name, output_dir, test_size=None):
    """
    Save evaluation metrics to a text file.
    """
    os.makedirs(output_dir, exist_ok=True)
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
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"{model_name}_architecture_{timestamp}.txt")
    
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

def save_training_parameters(optimizer, criterion, scheduler, output_dir):
    """
    Save details about the optimizer, criterion, and scheduler used for training.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"training_parameters_{timestamp}.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"=== Training Parameters ===\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Save criterion details
        f.write("=== Loss Function ===\n")
        f.write(f"Type: {criterion.__class__.__name__}\n")
        
        # Add specific criterion parameters based on the type
        if hasattr(criterion, 'alpha'):
            f.write(f"Alpha: {criterion.alpha}\n")
        if hasattr(criterion, 'gamma'):
            f.write(f"Gamma: {criterion.gamma}\n")
        if hasattr(criterion, 'beta'):
            f.write(f"Beta: {criterion.beta}\n")
        if hasattr(criterion, 'label_smoothing'):
            f.write(f"Label Smoothing: {criterion.label_smoothing}\n")
        if hasattr(criterion, 'num_classes'):
            f.write(f"Number of Classes: {criterion.num_classes}\n")
        if hasattr(criterion, 'focal_weight'):
            f.write(f"Focal Weight: {criterion.focal_weight}\n")
        if hasattr(criterion, 'wasserstein_weight'):
            f.write(f"Wasserstein Weight: {criterion.wasserstein_weight}\n")
        if hasattr(criterion, 'ordinal_weight'):
            f.write(f"Ordinal Weight: {criterion.ordinal_weight}\n")
            
        f.write("\n")
        
        # Save optimizer details
        f.write("=== Optimizer ===\n")
        f.write(f"Type: {optimizer.__class__.__name__}\n")
        f.write(f"Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"Weight Decay: {optimizer.param_groups[0]['weight_decay']}\n")
        if isinstance(optimizer, torch.optim.SGD):
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

            if hasattr(scheduler, 'mode'):
                f.write(f"Mode: {scheduler.mode}\n")
            if hasattr(scheduler, 'factor'):
                f.write(f"Factor: {scheduler.factor}\n")
            if hasattr(scheduler, 'patience'):
                f.write(f"Patience: {scheduler.patience}\n")
            if hasattr(scheduler, 'T_0'):
                f.write(f"T_0: {scheduler.T_0}\n")
            if hasattr(scheduler, 'T_mult'):
                f.write(f"T_mult: {scheduler.T_mult}\n")
            if hasattr(scheduler, 'eta_min'):
                f.write(f"eta_min: {scheduler.eta_min}\n")
                
    print(f"Training parameters saved to {file_path}")
    return file_path

def save_transformation_details(transforms_list, output_dir):
    """
    Save details about the image transformations used during training.
    """
    os.makedirs(output_dir, exist_ok=True)
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
            
            if hasattr(transform, 'degrees'):
                f.write(f"   - Degrees: {transform.degrees}\n")
            
            if hasattr(transform, 'brightness'):
                f.write(f"   - Brightness: {transform.brightness}\n")
            if hasattr(transform, 'contrast'):
                f.write(f"   - Contrast: {transform.contrast}\n")
            if hasattr(transform, 'saturation'):
                f.write(f"   - Saturation: {transform.saturation}\n")
            if hasattr(transform, 'hue'):
                f.write(f"   - Hue: {transform.hue}\n")
            
            if hasattr(transform, 'size'):
                f.write(f"   - Size: {transform.size}\n")
            
            if hasattr(transform, 'distortion_scale'):
                f.write(f"   - Distortion Scale: {transform.distortion_scale}\n")

            if hasattr(transform, 'scale'):
                f.write(f"   - Scale: {transform.scale}\n")
            
            if hasattr(transform, 'translate'):
                f.write(f"   - Translate: {transform.translate}\n")
            if hasattr(transform, 'shear'):
                f.write(f"   - Shear: {transform.shear}\n")
    
    print(f"Transformation details saved to {file_path}")
    return file_path

def plot_training_history(history, output_dir=None):
    """
    Plot training history metrics.
    """
    if output_dir is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
        
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
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()