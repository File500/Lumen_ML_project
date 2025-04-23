import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    balanced_accuracy_score, 
    precision_recall_fscore_support, 
    mean_absolute_error,
    mean_squared_error, 
    cohen_kappa_score, 
    roc_auc_score,
    roc_curve, 
    auc
)
from sklearn.preprocessing import label_binarize
from scipy.stats import spearmanr
from tqdm import tqdm

def distance_weighted_accuracy(y_true, y_pred, max_distance=9):
    """
    Calculate accuracy weighted by the distance between classes.
    Predictions that are closer to the true class receive higher scores.
    """
    distances = np.abs(y_pred - y_true)
    weights = 1.0 - (distances / max_distance)  # Higher weight for smaller distances
    return np.mean(weights)

def off_by_one_accuracy(y_true, y_pred):
    """
    Calculate the percentage of predictions that are at most one class 
    away from the true label.
    """
    return np.mean(np.abs(y_pred - y_true) <= 1)

def class_pair_confusion_rate(y_true, y_pred, class_a, class_b):
    """
    Calculate the confusion rate between a specific pair of classes.
    """
    # Filter samples where true label is either class_a or class_b
    mask = (y_true == class_a) | (y_true == class_b)
    if not np.any(mask):
        return 0.0
    
    # Get subset of true and predicted labels
    true_subset = y_true[mask]
    pred_subset = y_pred[mask]
    
    # Calculate confusion rate: samples where true and predicted classes don't match
    return np.mean(true_subset != pred_subset)

def calculate_similar_classes_confusion(y_true, y_pred, similar_classes=[0, 1, 2, 3]):
    """
    Calculate confusion rates between all pairs of similar classes.
    """
    confusion_rates = {}
    for i in similar_classes:
        for j in similar_classes:
            if i != j:
                rate = class_pair_confusion_rate(y_true, y_pred, i, j)
                confusion_rates[f"{i+1}_vs_{j+1}"] = rate
    return confusion_rates

def compute_ece(probs, labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    """
    # Get the confidence and predictions
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate ECE across bins
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # Calculate accuracy and average confidence in the bin
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            
            # Add weighted absolute difference to ECE
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
    
    return ece

def compute_brier_score(probs, labels, n_classes):
    """
    Compute Brier Score for multiclass classification.
    """
    # One-hot encode the labels
    y_true_one_hot = np.zeros((len(labels), n_classes))
    for i, label in enumerate(labels):
        y_true_one_hot[i, label] = 1
    
    # Calculate squared error between predictions and true one-hot vectors
    return np.mean(np.sum((probs - y_true_one_hot) ** 2, axis=1))

def plot_confusion_matrix(cm, class_names, output_dir=None):
    """
    Plot confusion matrix.
    """
    if output_dir is None:
        return
        
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_calibration_curve(probs, labels, output_dir=None, n_bins=10):
    """
    Plot a calibration curve to visualize model calibration.
    """
    if output_dir is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_counts.append(np.sum(in_bin))
        
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    
    # Add bin counts as bar chart in background
    ax2 = plt.gca().twinx()
    ax2.bar(bin_lowers + (bin_uppers - bin_lowers) / 2, bin_counts, 
            width=(bin_uppers - bin_lowers) * 0.8, alpha=0.3, color='gray', label='Samples per bin')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    plt.close()

class TemperatureScaledModel(nn.Module):
    """
    Wraps a model with temperature scaling for better calibration
    """
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = temperature
        
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature
    
def apply_temperature_scaling(model, val_loader, device):
    """
    Apply temperature scaling to a trained model
    """
    # Set model to eval mode
    model.eval()
    
    # Collect validation predictions
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            logits_list.append(logits)
            labels_list.append(labels)
    
    # Concatenate all batches
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    
    # Create temperature parameter - make sure it requires gradients
    temperature = nn.Parameter(torch.ones(1, requires_grad=True, device=device))
    
    # Define NLL loss criterion
    nll_criterion = nn.CrossEntropyLoss()
    
    # Define optimizer for temperature parameter only
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    
    # Define optimization step
    def eval_step():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = nll_criterion(scaled_logits, labels)
        loss.backward()
        return loss
    
    # Run optimization
    optimizer.step(eval_step)
    
    # Print optimal temperature
    print(f"Optimal temperature: {temperature.item():.4f}")
    
    # Return the calibrated model
    return TemperatureScaledModel(model, temperature.item())