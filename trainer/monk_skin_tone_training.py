import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.skin_tone_model import SkinToneClassifier, EfficientNetSkinToneClassifier, ResNetSkinToneClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

# Define image transformations for training
def get_transforms():
    return transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                     std=[0.229, 0.224, 0.225])
    ])

# Create dataset class


def train_model(training_data_path, output_model_path, output_best_model_path, image_folder):
    """
    Train a skin tone classification model from labeled data with train/validation/test split.
    
    Args:
        training_data_path: Path to training data with labeled skin types
        output_model_path: Path to save the trained model
        
    Returns:
        Trained model, compute device, and test dataframe
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    
    # Create model architecture
    #model = SkinToneClassifier(num_classes=10)
    model = EfficientNetSkinToneClassifier(num_classes=10)
    model.to(device)
    
    # Load data
    full_df = pd.read_csv(training_data_path)
    print(f"Loaded data with {len(full_df)} samples")
    
    if 'image_name' not in full_df.columns or 'predicted_skin_type' not in full_df.columns:
        print("Error: Data must have 'image_name' and 'predicted_skin_type' columns")
        return None, device, None
    
    # First split: separate out test set (20% of data)
    train_val_df, test_df = train_test_split(
        full_df, 
        test_size=0.2,
        random_state=42,
        stratify=full_df['predicted_skin_type']  # Ensure balanced classes
    )
    
    # Second split: divide remaining data into train and validation (80/20 split of the 80% remaining)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df['predicted_skin_type']
    )
    
    print(f"Split data into: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
    
    # Save the splits for reference
    split_dir = os.path.join(os.path.dirname(training_data_path), 'splits')
    os.makedirs(split_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(split_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(split_dir, 'validation_split.csv'), index=False)
    test_df.to_csv(os.path.join(split_dir, 'test_split.csv'), index=False)
    
    # Setup data transforms
    transform = get_transforms()
    
    # Create dataset class
    class SkinToneDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, transform=None, image_folder=None):
            self.dataframe = dataframe
            self.transform = transform
            self.image_folder = image_folder
        
        def __len__(self):
            return len(self.dataframe)
        
        def __getitem__(self, idx):
            image_name = self.dataframe.iloc[idx]['image_name']
            label = self.dataframe.iloc[idx]['predicted_skin_type'] - 1  # 0-indexed
            
            # Construct the full image path
            if self.image_folder:
                # Try different extensions
                for ext in ['.jpg', '.png', '.jpeg']:
                    img_path = os.path.join(self.image_folder, f"{image_name}{ext}")
                    if os.path.exists(img_path):
                        break
            else:
                # If no image folder provided, assume image_name already contains the path
                img_path = image_name
            
            # Check if image exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
    
    #Define image folder path
    
    print(f"Using images from: {image_folder}")

    # Create dataloaders with image folder parameter
    train_dataset = SkinToneDataset(train_df, transform, image_folder=image_folder)
    val_dataset = SkinToneDataset(val_df, transform, image_folder=image_folder)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    
    # Rest of the training function remains the same
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Prepare for training
    num_epochs = 25
    best_val_loss = float('inf')
    
    # Lists to track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f}')
        
        # Update scheduler
        scheduler.step(val_loss)

        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        torch.save(model.state_dict(), output_model_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_best_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_best_model_path)
            print(f"Saved improved model to {output_best_model_path}")
    
    # Load best model for return
    model.load_state_dict(torch.load(output_best_model_path))
    print("Training complete! Loaded best model.")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    
    # Save the plot
    plots_dir = os.path.join(os.path.dirname(output_best_model_path), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    return model, device, test_df

def predict_with_model(model, device, image_name, image_folder=None):
    """
    Predict skin tone using the trained model.
    
    Args:
        model: Trained model
        device: Compute device (CPU/GPU)
        image_name: Image name or path
        image_folder: Folder containing the images (if image_name is just a filename)
        
    Returns:
        Predicted Monk skin type (1-10)
    """
    try:
        # Construct the full image path if needed
        if image_folder:
            # Try different extensions
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = os.path.join(image_folder, f"{image_name}{ext}")
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            
            if image_path is None:
                raise FileNotFoundError(f"Image not found: {image_name}")
        else:
            # If no image folder provided, assume image_name already contains the path
            image_path = image_name
        
        # Open image with PIL
        img = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        transform = get_transforms()
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            
        # Return predicted skin type (1-10)
        return predicted.item() + 1  # +1 because model outputs 0-9
    
    except Exception as e:
        print(f"Error predicting for {image_name}: {e}")
        return None

def main():
    """Main function to run the script."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    image_folder = os.path.join(data_dir, 'train_224X224')
    training_data_path = os.path.join(data_dir, 'skin_type_analysis', 'clustering_comparison', 'spectral', 'ISIC_2020_with_monk_skin_types.csv')
    model_folder = os.path.join(project_root, 'trained_model')
    output_best_model_path = os.path.join(model_folder, 'best_monk_skin_tone_model.pth')
    output_model_path = os.path.join(model_folder, 'last_monk_skin_tone_model.pth')
    output_folder = os.path.join(project_root, 'model_evaluation', 'monk_skin_type')
    
    # Create directories if they don't exist
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Train the model and get the test set
    print(f"Training MobileNetV2 model using data from {training_data_path}")
    model, device, test_df = train_model(training_data_path, output_model_path, output_best_model_path, image_folder)
    
    if model is not None and test_df is not None:

        # Evaluate on the test set
        print("Evaluating model on test set...")
        image_folder = os.path.join(data_dir, 'train_224X224')
        test_predictions = []
        test_actual = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
            image_name = row['image_name']
            actual_type = row['predicted_skin_type']
            
            # Predict using the updated function with image folder
            pred_type = predict_with_model(model, device, image_name, image_folder=image_folder)
            
            if pred_type is not None:
                test_actual.append(actual_type)
                test_predictions.append(pred_type)
        
        # Convert to numpy arrays
        test_actual = np.array(test_actual)
        test_predictions = np.array(test_predictions)
        
        # Calculate accuracy
        test_accuracy = np.mean(test_actual == test_predictions)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate precision, recall, and F1 score (both macro and weighted)
        precision_macro = precision_score(test_actual, test_predictions, average='macro')
        precision_weighted = precision_score(test_actual, test_predictions, average='weighted')
        
        recall_macro = recall_score(test_actual, test_predictions, average='macro')
        recall_weighted = recall_score(test_actual, test_predictions, average='weighted')
        
        f1_macro = f1_score(test_actual, test_predictions, average='macro')
        f1_weighted = f1_score(test_actual, test_predictions, average='weighted')
        
        # Print comprehensive metrics
        print(f"Precision (macro): {precision_macro:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Recall (macro): {recall_macro:.4f}")
        print(f"Recall (weighted): {recall_weighted:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        
        # Get detailed classification report
        class_report = classification_report(test_actual, test_predictions)
        print("\nDetailed Classification Report:")
        print(class_report)
        
        # Save metrics to a text file
        with open(os.path.join(output_folder, 'test_metrics.txt'), 'w') as f:
            f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Precision (macro): {precision_macro:.4f}\n")
            f.write(f"Precision (weighted): {precision_weighted:.4f}\n")
            f.write(f"Recall (macro): {recall_macro:.4f}\n")
            f.write(f"Recall (weighted): {recall_weighted:.4f}\n")
            f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
            f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n\n")
            f.write("Detailed Classification Report:\n")
            f.write(class_report)
        
        # Create test confusion matrix
        cm = confusion_matrix(test_actual, test_predictions)
        
        # Plot test confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Test Set Confusion Matrix')
        plt.xticks(np.arange(10), list(range(1, 11)))
        plt.yticks(np.arange(10), list(range(1, 11)))
        
        # Add numbers to the confusion matrix
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > np.max(cm)/2 else 'black')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'test_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Create a DataFrame with test predictions
        test_results_df = test_df.copy()
        test_results_df['model_prediction'] = None

        # Add predictions to DataFrame
        for i, (idx, row) in enumerate(test_df.iterrows()):
            if i < len(test_predictions):
                test_results_df.loc[idx, 'model_prediction'] = test_predictions[i]

        # Save test predictions to CSV
        test_results_path = os.path.join(output_folder, 'test_predictions.csv')
        test_results_df.to_csv(test_results_path, index=False)
        print(f"Test predictions saved to {test_results_path}")
        
        print(f"MobileNetV2 model training and evaluation complete! Model saved to {output_model_path}")
    else:
        print("Model training failed.")

if __name__ == "__main__":
    main()