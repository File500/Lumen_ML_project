import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shutil
from datetime import datetime
import seaborn as sns

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import EfficientNetB3SkinToneClassifier
from model.modified_melanoma_model import ModifiedMelanomaClassifier
from model.combined_model import CombinedTransferModel

# Dataset class for loading single images
class MelanomaImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_dir, transform=None, binary_mode=True):
        self.data_frame = dataframe
        self.img_dir = img_dir
        self.binary_mode = binary_mode
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data_frame.iloc[idx, 0]  # Assuming first column is image name
        
        # Check if extension is missing and add .jpg if needed
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_name = img_name + '.jpg'
        
        # Create full path
        img_path = os.path.join(self.img_dir, img_name)
        
        # Try to open the image
        try:
            image = Image.open(img_path).convert('RGB')
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
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, target, img_name

def get_device():
    """Get the best available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_combined_model(skin_model_path, melanoma_model_path, combined_model_path):
    """Load all models needed for combined prediction"""
    device = get_device()
    
    # 1. Load skin tone classifier
    print(f"Loading skin tone model from {skin_model_path}")
    skin_model = EfficientNetB3SkinToneClassifier(num_classes=7)
    skin_state_dict = torch.load(skin_model_path, map_location=device)
    
    # Check and load correct format for skin model
    if isinstance(skin_state_dict, dict):
        if 'model_state' in skin_state_dict:
            state_dict = skin_state_dict['model_state']
        elif 'state_dict' in skin_state_dict:
            state_dict = skin_state_dict['state_dict']
        elif 'model_state_dict' in skin_state_dict:
            state_dict = skin_state_dict['model_state_dict']
        else:
            state_dict = skin_state_dict
    else:
        state_dict = skin_state_dict
    
    skin_model.load_state_dict(state_dict)
    skin_model.to(device)
    skin_model.eval()
    
    # 2. Load melanoma model
    print(f"Loading melanoma model from {melanoma_model_path}")
    melanoma_model = ModifiedMelanomaClassifier(num_classes=2, binary_mode=True)
    melanoma_state_dict = torch.load(melanoma_model_path, map_location=device)
    
    # Check and load correct format for melanoma model
    if 'model_state' in melanoma_state_dict:
        melanoma_model.load_state_dict(melanoma_state_dict['model_state'])
    elif 'model_state_dict' in melanoma_state_dict:
        melanoma_model.load_state_dict(melanoma_state_dict['model_state_dict'])
    else:
        melanoma_model.load_state_dict(melanoma_state_dict)
    
    melanoma_model.to(device)
    melanoma_model.eval()
    
    # 3. Load combined model
    print(f"Loading combined model from {combined_model_path}")
    combined_model = CombinedTransferModel(
        skin_tone_model=skin_model,
        melanoma_model=melanoma_model,
        num_classes=2,
        binary_mode=True
    )
    
    combined_state_dict = torch.load(combined_model_path, map_location=device)
    
    # Check and load correct format for combined model
    if 'model_state' in combined_state_dict:
        combined_model.load_state_dict(combined_state_dict['model_state'])
    elif 'model_state_dict' in combined_state_dict:
        combined_model.load_state_dict(combined_state_dict['model_state_dict'])
    else:
        combined_model.load_state_dict(combined_state_dict)
    
    combined_model.to(device)
    combined_model.eval()
    
    return combined_model

def predict_on_test_set(model, test_df, img_dir, threshold=0.3, save_misclassified=True):
    """Run predictions on the test set and save misclassified images"""
    device = get_device()
    model.eval()
    
    # Create transforms for evaluation
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    test_dataset = MelanomaImageDataset(
        dataframe=test_df,
        img_dir=img_dir,
        transform=eval_transform,
        binary_mode=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create output directories for misclassified images
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("prediction_results", f"results_{timestamp}")
    
    # Create separate directories for false positives and false negatives
    fp_dir = os.path.join(output_dir, "false_positives")
    fn_dir = os.path.join(output_dir, "false_negatives")
    
    if save_misclassified:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fp_dir, exist_ok=True)
        os.makedirs(fn_dir, exist_ok=True)
        
        # Create directories for the montage images
        os.makedirs(os.path.join(output_dir, "montages"), exist_ok=True)
    
    # Lists to store results
    all_preds = []
    all_probs = []
    all_labels = []
    false_positives = []
    false_negatives = []
    
    # Run predictions
    print("Running predictions on test set...")
    with torch.no_grad():
        for images, labels, img_names in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            # Convert to numpy for storage
            probs_np = probs.cpu().numpy()
            preds_np = (probs_np > threshold).astype(float)
            labels_np = labels.cpu().numpy()
            
            # Store predictions and labels
            all_probs.extend(probs_np.flatten())
            all_preds.extend(preds_np.flatten())
            all_labels.extend(labels_np.flatten())
            
            # Check for misclassified images
            if save_misclassified:
                for i in range(len(images)):
                    img_name = img_names[i]
                    true_label = labels_np[i][0]
                    pred_label = preds_np[i][0]
                    prob_value = probs_np[i][0]
                    
                    # False positive: predicted malignant (1) but actually benign (0)
                    if pred_label == 1 and true_label == 0:
                        false_positives.append((img_name, prob_value))
                        
                        # Copy the image to false positives directory
                        src_path = os.path.join(img_dir, img_name)
                        if not os.path.isfile(src_path):
                            src_path = os.path.join(img_dir, img_name + '.jpg')
                        
                        if os.path.isfile(src_path):
                            dest_path = os.path.join(fp_dir, f"{img_name}_prob_{prob_value:.3f}.jpg")
                            shutil.copy2(src_path, dest_path)
                    
                    # False negative: predicted benign (0) but actually malignant (1)
                    elif pred_label == 0 and true_label == 1:
                        false_negatives.append((img_name, prob_value))
                        
                        # Copy the image to false negatives directory
                        src_path = os.path.join(img_dir, img_name)
                        if not os.path.isfile(src_path):
                            src_path = os.path.join(img_dir, img_name + '.jpg')
                        
                        if os.path.isfile(src_path):
                            dest_path = os.path.join(fn_dir, f"{img_name}_prob_{prob_value:.3f}.jpg")
                            shutil.copy2(src_path, dest_path)
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Print metrics
    print("\n==== Test Set Metrics ====")
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    if save_misclassified:
        # Save metrics to a text file
        metrics_file = os.path.join(output_dir, "metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"Test Set Metrics (Threshold: {threshold})\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(f"True Negatives: {tn}\n")
            f.write(f"False Positives: {fp}\n")
            f.write(f"False Negatives: {fn}\n")
            f.write(f"True Positives: {tp}\n\n")
            f.write(f"Total False Positives: {len(false_positives)}\n")
            f.write(f"Total False Negatives: {len(false_negatives)}\n")
        
        # Create montage of false positives (if any)
        if false_positives:
            create_image_montage(
                img_dir=fp_dir, 
                output_path=os.path.join(output_dir, "montages", "false_positives_montage.png"),
                title="False Positives (Benign lesions classified as Malignant)"
            )

        # Create montage of false negatives (if any)
        if false_negatives:
            create_image_montage(
                img_dir=fn_dir, 
                output_path=os.path.join(output_dir, "montages", "false_negatives_montage.png"),
                title="False Negatives (Malignant lesions classified as Benign)"
            )
        
        print(f"\nResults saved to {output_dir}")
        print(f"- False Positives: {len(false_positives)} images")
        print(f"- False Negatives: {len(false_negatives)} images")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'predictions': all_preds,  # Add this line
        'probabilities': all_probs  # Add this line
    }

def analyze_predictions_by_skin_type(test_df, predictions, probabilities, output_dir):
    """
    Analyze existing predictions by skin type
    
    Args:
        test_df (pd.DataFrame): DataFrame containing the test data with monk_skin_type column
        predictions (list/array): Binary predictions (0 or 1)
        probabilities (list/array): Prediction probabilities
        output_dir (str): Directory to save visualizations
    
    Returns:
        pd.DataFrame: DataFrame with metrics by skin type
    """
    # Convert inputs to numpy arrays if they aren't already
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Ensure test_df has the same number of rows as predictions
    if len(test_df) != len(predictions):
        raise ValueError(f"DataFrame has {len(test_df)} rows but predictions array has {len(predictions)} elements")
    
    # Create a result DataFrame
    result_df = test_df.copy()
    result_df['prediction'] = predictions
    result_df['probability'] = probabilities
    result_df['correct'] = (result_df['prediction'] == result_df['target']).astype(int)
    
    # Group by skin type
    skin_types = sorted(result_df['monk_skin_type'].unique())
    metrics_data = []
    
    print("Analyzing performance by skin type...")
    
    # Calculate metrics for each skin type
    for skin_type in skin_types:
        group = result_df[result_df['monk_skin_type'] == skin_type]
        
        # Skip if group is empty
        if len(group) == 0:
            continue
        
        # Get actual labels and predictions for this group
        y_true = group['target'].values
        y_pred = group['prediction'].values
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate error rates
        total_negatives = tn + fp
        total_positives = tp + fn
        
        fpr = fp / total_negatives if total_negatives > 0 else 0
        fnr = fn / total_positives if total_positives > 0 else 0
        
        # Calculate class distribution
        benign_count = sum(y_true == 0)
        malignant_count = sum(y_true == 1)
        benign_percent = benign_count / len(group) * 100
        malignant_percent = malignant_count / len(group) * 100
        
        # Store metrics
        metrics_data.append({
            'skin_type': skin_type,
            'count': len(group),
            'benign_count': benign_count,
            'malignant_count': malignant_count,
            'benign_percent': benign_percent,
            'malignant_percent': malignant_percent,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr
        })
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(metrics_data)
    
    # Set skin_type as index
    metrics_df.set_index('skin_type', inplace=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the metrics DataFrame to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'skin_type_metrics.csv'))
    
    # Create visualizations
    
    # 1. Accuracy by skin type
    plt.figure(figsize=(12, 6))
    ax = metrics_df['accuracy'].plot(kind='bar', color='skyblue')
    plt.xlabel('Monk Skin Type')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy by Skin Type')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of bars
    for i, v in enumerate(metrics_df['accuracy']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Add sample count at the bottom of each bar
    for i, v in enumerate(metrics_df['count']):
        ax.text(i, 0.02, f'n={v}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_skin_type.png'), dpi=300)
    plt.close()
    
    # 2. Multiple metrics comparison
    plt.figure(figsize=(14, 7))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create a grouped bar chart
    bar_width = 0.2
    index = np.arange(len(skin_types))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(index + i*bar_width - bar_width*1.5, metrics_df[metric], width=bar_width, label=metric.capitalize())
    
    plt.xlabel('Monk Skin Type')
    plt.ylabel('Score')
    plt.title('Prediction Metrics by Skin Type')
    plt.xticks(index, skin_types)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_by_skin_type.png'), dpi=300)
    plt.close()
    
    # 3. Error rates heatmap
    plt.figure(figsize=(12, 6))
    error_df = pd.DataFrame({
        'False Positive Rate': metrics_df['false_positive_rate'],
        'False Negative Rate': metrics_df['false_negative_rate']
    })
    
    sns.heatmap(error_df.T, annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=0.5)
    plt.title('Error Rates by Skin Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_rates_by_skin_type.png'), dpi=300)
    plt.close()
    
    # 4. Class distribution by skin type
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(skin_types))
    
    plt.bar(index - bar_width/2, metrics_df['benign_percent'], bar_width, label='Benign', color='green', alpha=0.7)
    plt.bar(index + bar_width/2, metrics_df['malignant_percent'], bar_width, label='Malignant', color='red', alpha=0.7)
    
    plt.xlabel('Monk Skin Type')
    plt.ylabel('Percentage (%)')
    plt.title('Class Distribution by Skin Type')
    plt.xticks(index, skin_types)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add percentage values on top of bars
    for i, v in enumerate(metrics_df['benign_percent']):
        plt.text(i - bar_width/2, v + 1, f'{v:.1f}%', ha='center')
    
    for i, v in enumerate(metrics_df['malignant_percent']):
        plt.text(i + bar_width/2, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_by_skin_type.png'), dpi=300)
    plt.close()
    
    # Create a detailed report
    report_path = os.path.join(output_dir, 'skin_type_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("===== SKIN TYPE ANALYSIS REPORT =====\n\n")
        
        # Overall summary
        f.write("OVERALL SUMMARY:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total samples analyzed: {sum(metrics_df['count'])}\n")
        f.write(f"Number of skin types: {len(metrics_df)}\n\n")
        
        # Best and worst performing skin types
        best_skin_type = metrics_df['accuracy'].idxmax()
        worst_skin_type = metrics_df['accuracy'].idxmin()
        
        f.write(f"Best performing skin type: {best_skin_type} (Accuracy: {metrics_df.loc[best_skin_type, 'accuracy']:.4f})\n")
        f.write(f"Worst performing skin type: {worst_skin_type} (Accuracy: {metrics_df.loc[worst_skin_type, 'accuracy']:.4f})\n\n")
        
        # Skin type with highest false positive rate
        highest_fpr_type = metrics_df['false_positive_rate'].idxmax()
        f.write(f"Skin type with highest false positive rate: {highest_fpr_type} (Rate: {metrics_df.loc[highest_fpr_type, 'false_positive_rate']:.4f})\n")
        
        # Skin type with highest false negative rate
        highest_fnr_type = metrics_df['false_negative_rate'].idxmax()
        f.write(f"Skin type with highest false negative rate: {highest_fnr_type} (Rate: {metrics_df.loc[highest_fnr_type, 'false_negative_rate']:.4f})\n\n")
        
        # Detailed metrics by skin type
        f.write("DETAILED METRICS BY SKIN TYPE:\n")
        f.write("-" * 50 + "\n")
        
        for skin_type in metrics_df.index:
            metrics = metrics_df.loc[skin_type]
            f.write(f"SKIN TYPE {skin_type}:\n")
            f.write(f"  Sample count: {metrics['count']}\n")
            f.write(f"  Class distribution: {metrics['benign_count']} benign ({metrics['benign_percent']:.1f}%), "
                    f"{metrics['malignant_count']} malignant ({metrics['malignant_percent']:.1f}%)\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  Confusion Matrix: [TN={metrics['true_negatives']}, FP={metrics['false_positives']}, "
                    f"FN={metrics['false_negatives']}, TP={metrics['true_positives']}]\n")
            f.write(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}\n")
            f.write(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}\n\n")
    
    print(f"Skin type analysis complete. Results saved to {output_dir}")
    print(f"- Best performing skin type: {best_skin_type} (Accuracy: {metrics_df.loc[best_skin_type, 'accuracy']:.4f})")
    print(f"- Worst performing skin type: {worst_skin_type} (Accuracy: {metrics_df.loc[worst_skin_type, 'accuracy']:.4f})")
    
    return metrics_df

def create_image_montage(img_dir, output_path, title=None):
    """Create a montage of ALL images from the directory"""
    if not os.path.exists(img_dir) or not os.listdir(img_dir):
        print(f"No images found in {img_dir}")
        return
    
    # Get all image files
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_files)
    
    # Calculate grid dimensions - make grid wider than tall for better viewing
    # This creates a grid that's approximately 1:2 (height:width) ratio
    grid_width = int(np.ceil(np.sqrt(num_images * 2)))
    grid_height = int(np.ceil(num_images / grid_width))
    
    print(f"Creating montage with {num_images} images in a {grid_height}x{grid_width} grid")
    
    # Create figure for montage - adjust figure size based on number of images
    fig_width = min(30, grid_width * 2)  # Limit max width
    fig_height = min(20, grid_height * 2)  # Limit max height
    
    fig, axes = plt.subplots(grid_height, grid_width, figsize=(fig_width, fig_height))
    
    # Make axes a 2D array even if it's 1D
    if grid_height == 1 and grid_width == 1:
        axes = np.array([[axes]])
    elif grid_height == 1:
        axes = axes.reshape(1, -1)
    elif grid_width == 1:
        axes = axes.reshape(-1, 1)
    
    # Loop through all images
    for i, img_file in enumerate(image_files):
        row = i // grid_width
        col = i % grid_width
        
        # Read image
        img_path = os.path.join(img_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        
        # Extract probability from filename (if it exists)
        prob_text = ""
        if "_prob_" in img_file:
            prob_str = img_file.split("_prob_")[1].split(".")[0]
            try:
                prob = float(prob_str)
                prob_text = f"Prob: {prob:.3f}"
            except:
                pass
        
        # Display image
        ax = axes[row, col]
        ax.imshow(img)
        
        # Use smaller font for larger grids
        fontsize = max(4, min(8, int(12 - np.log(num_images))))
        ax.set_title(f"{img_file.split('_prob_')[0]}\n{prob_text}", fontsize=fontsize)
        ax.axis('off')
    
    # Turn off extra subplot axes
    for i in range(num_images, grid_height * grid_width):
        row = i // grid_width
        col = i % grid_width
        axes[row, col].axis('off')
    
    # Set overall title
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save the montage - use higher DPI for larger image sets
    dpi = min(300, max(150, int(600 / np.sqrt(num_images))))
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Montage with {num_images} images saved to {output_path}")

def main():
    """Main function to run the predictions"""
    # Configuration
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    PROJECT_PATH = os.path.dirname(CURRENT_DIR)

    DATA_PATH = os.path.join(PROJECT_PATH, "data")

    config = {
        # Paths to models and data
        "skin_model_path": os.path.join(PROJECT_PATH, "trained_model", "skin_type_classifier", "EFNet_b3_300X300_final", "final_model.pth"),
        "melanoma_model_path":  os.path.join(PROJECT_PATH, "trained_model", "feature_extractor_melanoma_classifier_final", "modified_melanoma_model.pth"), 
        "combined_model_path":  os.path.join(PROJECT_PATH, "trained_model", "combined_model_final", "best_combined_model.pth"),
        
        # Path to test data
        "test_csv_path": os.path.join(DATA_PATH, "dataset_splits", "combined_test_split.csv"),
        "img_dir": os.path.join(DATA_PATH, "train_300X300_processed") ,
        
        # Prediction threshold (adjust as needed)
        "threshold": 0.30,
        
        # Whether to save misclassified images
        "save_misclassified": True
    }
    
    # Check if files exist
    for key in ["skin_model_path", "melanoma_model_path", "combined_model_path", "test_csv_path"]:
        if not os.path.exists(config[key]):
            print(f"ERROR: {key} not found at {config[key]}")
            return
    
    if not os.path.exists(config["img_dir"]):
        print(f"ERROR: Image directory not found at {config['img_dir']}")
        return
    
    # Load test set
    print(f"Loading test set from {config['test_csv_path']}")
    test_df = pd.read_csv(config["test_csv_path"])
    print(f"Loaded {len(test_df)} test samples")
    
    # Load models
    model = load_combined_model(
        skin_model_path=config["skin_model_path"],
        melanoma_model_path=config["melanoma_model_path"],
        combined_model_path=config["combined_model_path"]
    )
    
    # Run predictions
    results = predict_on_test_set(
        model=model,
        test_df=test_df,
        img_dir=config["img_dir"],
        threshold=config["threshold"],
        save_misclassified=config["save_misclassified"]
    )

    skin_analysis_dir = os.path.join("prediction_results", "skin_type_analysis")
    os.makedirs(skin_analysis_dir, exist_ok=True)

    # Run the skin type analysis
    skin_metrics = analyze_predictions_by_skin_type(
        test_df=test_df,
        predictions=results['predictions'],  
        probabilities=results['probabilities'], 
        output_dir=skin_analysis_dir
    )

if __name__ == "__main__":
    main()