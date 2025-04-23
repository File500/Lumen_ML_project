import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc, 
    confusion_matrix
)

def compute_classification_metrics(labels, predictions, threshold=0.3):
    """
    Compute classification metrics.
    
    Args:
        labels (np.ndarray or torch.Tensor): True labels
        predictions (np.ndarray or torch.Tensor): Model predictions
        threshold (float): Probability threshold for classification
    
    Returns:
        dict: Classification metrics
    """
    # Convert to numpy if tensor
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Flatten and convert to binary
    labels = labels.flatten()
    preds = (predictions > threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0, pos_label=1),
        'recall': recall_score(labels, preds, zero_division=0, pos_label=1),
        'f1_score': f1_score(labels, preds, zero_division=0, pos_label=1)
    }
    
    return metrics

def plot_roc_curve(labels, predictions, save_path=None):
    """
    Plot ROC curve and compute AUC.
    
    Args:
        labels (np.ndarray or torch.Tensor): True labels
        predictions (np.ndarray or torch.Tensor): Model predictions
        save_path (str, optional): Path to save the ROC curve plot
    
    Returns:
        float: ROC AUC score
    """
    # Convert to numpy if tensor
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Flatten
    labels = labels.flatten()
    predictions = predictions.flatten()
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions)
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
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    return roc_auc

def plot_confusion_matrix(labels, predictions, save_path=None, threshold=0.3):
    """
    Plot confusion matrix.
    
    Args:
        labels (np.ndarray or torch.Tensor): True labels
        predictions (np.ndarray or torch.Tensor): Model predictions
        save_path (str, optional): Path to save the confusion matrix plot
        threshold (float): Probability threshold for classification
    
    Returns:
        np.ndarray: Confusion matrix
    """
    # Convert to numpy if tensor
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    # Flatten and convert to binary
    labels = labels.flatten()
    preds = (predictions > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'],
                cbar=True)
    
    plt.title('Confusion Matrix', fontsize=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add explanatory text
    tn, fp, fn, tp = cm.ravel()
    plt.text(1.1, -0.1, 
            f'True Negatives: {tn}\n'
            f'False Positives: {fp}\n'
            f'False Negatives: {fn}\n'
            f'True Positives: {tp}', 
            horizontalalignment='left',
            verticalalignment='center',
            transform=plt.gca().transAxes)
    
    # Save plot if path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return cm