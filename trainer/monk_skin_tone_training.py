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
from model.skin_tone_model import SkinToneClassifier

# Define image transformations for training
def get_transforms():
    return transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                     std=[0.229, 0.224, 0.225])
    ])


def train_model(training_data_path, output_model_path):
    """
    Train a skin tone classification model from labeled data.
    
    Args:
        training_data_path: Path to training data with labeled skin types
        output_model_path: Path to save the trained model
        
    Returns:
        Trained model and compute device
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for training")
    
    # Create model architecture
    model = SkinToneClassifier(num_classes=10)
    model.to(device)
    
    # Load training data
    train_df = pd.read_csv(training_data_path)
    print(f"Loaded training data with {len(train_df)} samples")
    
    if 'image_path' not in train_df.columns or 'monk_skin_type' not in train_df.columns:
        print("Error: Training data must have 'image_path' and 'monk_skin_type' columns")
        return None, device
    
    # Setup data transforms
    transform = get_transforms()
    
    # Create dataset
    class SkinToneDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform
        
        def __len__(self):
            return len(self.dataframe)
        
        def __getitem__(self, idx):
            img_path = self.dataframe.iloc[idx]['image_path']
            label = self.dataframe.iloc[idx]['monk_skin_type'] - 1  # 0-indexed
            
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, label
    
    # Split data into training and validation sets
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_data)} samples, Validation set: {len(val_data)} samples")
    
    # Create dataloaders
    train_dataset = SkinToneDataset(train_data, transform)
    val_dataset = SkinToneDataset(val_data, transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Prepare for training
    num_epochs = 20
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            torch.save(model.state_dict(), output_model_path)
            print(f"Saved improved model to {output_model_path}")
    
    # Load best model for return
    model.load_state_dict(torch.load(output_model_path))
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
    plots_dir = os.path.join(os.path.dirname(output_model_path), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    return model, device

def predict_with_model(model, device, image_path):
    """
    Predict skin tone using the trained model.
    
    Args:
        model: Trained model
        device: Compute device (CPU/GPU)
        image_path: Path to the image
        
    Returns:
        Predicted Monk skin type (1-10)
    """
    try:
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
        print(f"Error predicting for {image_path}: {e}")
        return None

def visualize_training_results(training_data_path, model, device, output_folder):
    """
    Visualize the model's predictions on the training data.
    
    Args:
        training_data_path: Path to the training data
        model: Trained model
        device: Compute device
        output_folder: Folder to save visualizations
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load training data
    train_df = pd.read_csv(training_data_path)
    
    # Create a confusion matrix
    actual = []
    predicted = []
    
    # Make predictions
    print("Evaluating model on training data...")
    for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_path = row['image_path']
        actual_type = row['monk_skin_type']
        
        # Predict
        pred_type = predict_with_model(model, device, image_path)
        
        if pred_type is not None:
            actual.append(actual_type)
            predicted.append(pred_type)
    
    # Convert to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Calculate accuracy
    accuracy = np.mean(actual == predicted)
    print(f"Accuracy on training data: {accuracy:.4f}")
    
    # Create confusion matrix
    confusion = np.zeros((10, 10), dtype=int)
    for a, p in zip(actual, predicted):
        confusion[a-1, p-1] += 1
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion, cmap='Blues')
    plt.colorbar()
    
    # Add labels
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add ticks
    tick_labels = list(range(1, 11))
    plt.xticks(np.arange(10), tick_labels)
    plt.yticks(np.arange(10), tick_labels)
    
    # Add numbers
    for i in range(10):
        for j in range(10):
            plt.text(j, i, confusion[i, j], ha='center', va='center', 
                     color='white' if confusion[i, j] > np.max(confusion)/2 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    pd.Series(actual).value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title('Actual Skin Type Distribution')
    plt.xlabel('Monk Skin Type')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    pd.Series(predicted).value_counts().sort_index().plot(kind='bar', color='salmon')
    plt.title('Predicted Skin Type Distribution')
    plt.xlabel('Monk Skin Type')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'distribution_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main function to run the script."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    training_data_path = os.path.join(data_dir, 'labeled_skin_types.csv')
    model_folder = os.path.join(project_root, 'trained_model')
    output_model_path = os.path.join(model_folder, 'monk_skin_tone_model.pth')
    output_folder = os.path.join(data_dir, 'model_evaluation')
    
    # Create directories if they don't exist
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Train the model
    print(f"Training model using data from {training_data_path}")
    model, device = train_model(training_data_path, output_model_path)
    
    if model is not None:
        # Visualize training results
        visualize_training_results(training_data_path, model, device, output_folder)
        print(f"Model training complete! Model saved to {output_model_path}")
    else:
        print("Model training failed.")

if __name__ == "__main__":
    main()