import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Define image transformations for the pre-trained model
def get_transforms():
    return transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                     std=[0.229, 0.224, 0.225])
    ])

class SkinToneClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(SkinToneClassifier, self).__init__()
        # Use a pre-trained ResNet as the backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_pretrained_model(model_path=None):
    """
    Load a pre-trained skin tone classification model or download one.
    
    Args:
        model_path: Path to pre-trained model if available
    
    Returns:
        Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkinToneClassifier(num_classes=10)
    
    if model_path and os.path.exists(model_path):
        # Load local model
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No pre-trained model provided. Using feature extraction method.")
        model = None
    
    if model:
        model.to(device)
        model.eval()
    
    return model, device

def extract_skin_features(image_path):
    """
    Extract features related to skin tone from the image.
    Automatically crops black lines from top and bottom.
    
    Args:
        image_path: Path to the dermoscopic image
        
    Returns:
        Dictionary of features related to skin color/tone
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        # Crop black lines from top and bottom
        img = crop_black_lines(img)
        
        # Convert to LAB color space (better for skin tone analysis)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Extract L (lightness), A (red-green), B (blue-yellow) channels
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        # Use KMeans to find dominant colors (5 clusters)
        pixels = lab_img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the dominant color values and sort by their frequency
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Sort colors by count (most frequent first)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = colors[sorted_indices]
        
        # Calculate color features - we use the top 3 dominant colors
        features = {}
        
        # Global image statistics
        features['avg_l'] = np.mean(l_channel)
        features['std_l'] = np.std(l_channel)
        features['avg_a'] = np.mean(a_channel)
        features['std_a'] = np.std(a_channel)
        features['avg_b'] = np.mean(b_channel)
        features['std_b'] = np.std(b_channel)
        
        # Dominant color features (top 3)
        for i in range(min(3, len(sorted_colors))):
            features[f'dom{i+1}_l'] = sorted_colors[i][0]
            features[f'dom{i+1}_a'] = sorted_colors[i][1]
            features[f'dom{i+1}_b'] = sorted_colors[i][2]
            features[f'dom{i+1}_freq'] = counts[sorted_indices[i]] / len(labels)
        
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
def extract_skin_features_advanced(image_path):
    """
    Extract features related to skin tone from the image.
    Uses both cropping and masking to focus on the skin region.
    
    Args:
        image_path: Path to the dermoscopic image
        
    Returns:
        Dictionary of features related to skin color/tone
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        
        if img is None:
            return None
        
        # Crop black lines from top and bottom
        img = crop_black_lines(img)
        
        # Create mask to exclude very dark areas (likely not skin)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray > 30  # Threshold to exclude near-black pixels
        
        # Convert to LAB color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Extract L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        # Apply mask to color channels
        l_masked = l_channel[mask]
        a_masked = a_channel[mask]
        b_masked = b_channel[mask]
        
        # If mask removes too much of the image, fallback to using the whole image
        if len(l_masked) < (l_channel.size * 0.1):  # If less than 10% remains
            print(f"Warning: Mask too aggressive for {image_path}, using full image")
            l_masked, a_masked, b_masked = l_channel, a_channel, b_channel
        
        # Use KMeans to find dominant colors (5 clusters) from masked pixels
        pixels = np.column_stack((l_masked, a_masked, b_masked))
        
        # Handle case where there are fewer pixels than clusters
        n_clusters = min(5, len(pixels))
        if n_clusters < 2:
            print(f"Warning: Not enough pixels for clustering in {image_path}")
            return None
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get the dominant color values and sort by their frequency
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Sort colors by count (most frequent first)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = colors[sorted_indices]
        
        # Calculate color features
        features = {}
        
        # Global image statistics (masked)
        features['avg_l'] = np.mean(l_masked)
        features['std_l'] = np.std(l_masked)
        features['avg_a'] = np.mean(a_masked)
        features['std_a'] = np.std(a_masked)
        features['avg_b'] = np.mean(b_masked)
        features['std_b'] = np.std(b_masked)
        
        # Dominant color features (top 3 or fewer)
        for i in range(min(3, len(sorted_colors))):
            features[f'dom{i+1}_l'] = sorted_colors[i][0]
            features[f'dom{i+1}_a'] = sorted_colors[i][1]
            features[f'dom{i+1}_b'] = sorted_colors[i][2]
            features[f'dom{i+1}_freq'] = counts[sorted_indices[i]] / len(labels)
        
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def crop_black_lines(img):
    """
    Crop black lines from top and bottom of an image.
    
    Args:
        img: Image to crop
        
    Returns:
        Cropped image
    """
    if img is None or img.size == 0:
        return img
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image height and width
    height, width = gray.shape
    
    # Find top crop line
    top_line = 0
    for i in range(height):
        # Check if the row has non-black pixels (use a threshold to account for noise)
        if np.mean(gray[i, :]) > 10:  # Threshold of 10 for near-black
            top_line = i
            break
    
    # Find bottom crop line
    bottom_line = height - 1
    for i in range(height - 1, -1, -1):
        if np.mean(gray[i, :]) > 10:
            bottom_line = i
            break
    
    # If entire image is black or cropping would remove everything, return original
    if top_line >= bottom_line:
        return img
    
    # Crop the image
    cropped_img = img[top_line:bottom_line + 1, :]
    
    return cropped_img

def predict_with_pretrained_model(model, device, image_path, transform):
    """
    Predict skin tone using a pre-trained model.
    
    Args:
        model: Pre-trained model
        device: Compute device (CPU/GPU)
        image_path: Path to the image
        transform: Image transformations
        
    Returns:
        Predicted Monk skin type (1-10)
    """
    try:
        # Open image with PIL
        img = Image.open(image_path).convert('RGB')
        
        # Apply transforms
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

def classify_with_clustering(features_df):
    """
    Classify skin types using clustering when no pre-trained model is available.
    This approach maps to the 10 skin types of the Monk scale.
    
    Args:
        features_df: DataFrame with extracted skin features
        
    Returns:
        DataFrame with added skin type predictions
    """
    # Features to use for clustering
    feature_cols = [col for col in features_df.columns if col.startswith(('avg_', 'std_', 'dom'))]
    
    # Create feature matrix
    X = features_df[feature_cols].copy()
    
    # Normalize features
    X = (X - X.mean()) / X.std()
    
    # Use KMeans to cluster the data into 10 clusters (for 10 Monk skin types)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Get cluster assignments
    clusters = kmeans.labels_
    
    # Map clusters to skin types based on color characteristics
    cluster_centers = kmeans.cluster_centers_
    
    # Find average L value (lightness) for each cluster
    l_index = feature_cols.index('avg_l')
    a_index = feature_cols.index('avg_a')
    b_index = feature_cols.index('avg_b')
    
    # Calculate a combined skin tone score using L, a, b values
    # Lower L = darker skin, higher a = more red, higher b = more yellow
    # This is a simplistic approach - actual skin tone mapping would need more refinement
    skin_tone_scores = []
    for center in cluster_centers:
        # Invert L value so higher value = darker skin
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]         # a can be negative (green) or positive (red)
        b_value = center[b_index]         # b can be negative (blue) or positive (yellow)
        
        # Simple weighted score - this is approximate and would need refinement
        # for more accurate Monk scale mapping
        score = (0.7 * l_value) + (0.15 * a_value) + (0.15 * b_value)
        skin_tone_scores.append(score)
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    sorted_indices = np.argsort(skin_tone_scores)
    
    # Map original clusters to Monk skin types (1-10)
    cluster_to_skin_type = {}
    for skin_type, cluster in enumerate(sorted_indices, 1):
        cluster_to_skin_type[cluster] = skin_type
    
    # Apply mapping to get predicted skin types
    features_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in clusters]
    
    return features_df

def process_dataset(csv_path, image_folder, output_folder, max_images=None):
    """
    Process the ISIC dataset and classify images using the Monk skin type scale.
    
    Parameters:
        csv_path: Path to the ISIC 2020 metadata CSV
        image_folder: Path to the folder containing the images
        output_folder: Folder to save results
        max_images: Maximum number of images to process (None for all)
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the metadata
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata with {len(df)} entries")
    
    # Limit to max_images if specified
    if max_images is not None:
        df = df.head(max_images)
        print(f"Processing first {len(df)} images")
    
    # Create a list to store results
    results = []
    
    # Process each image
    print("Processing images...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']
        
        # Try to find the image file
        image_path = os.path.join(image_folder, f"{image_name}.jpg")
        
        # Try different extensions if not found
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(image_folder, f"{image_name}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
            
            # Skip if image still not found
            if not os.path.exists(image_path):
                print(f"Image not found: {image_name}")
                continue
        
        # Extract features
        features = extract_skin_features_advanced(image_path)
        
        if features:
            # Add image name to features dictionary
            features['image_name'] = image_name
            results.append(features)
    
    # Convert results to DataFrame
    if not results:
        print("No images could be processed.")
        return None
    
    results_df = pd.DataFrame(results)
    print(f"Successfully processed {len(results_df)} images")
    
    # Classify skin types
    if len(results_df) > 0:
        results_df = classify_with_clustering(results_df)
        
        # Merge with original metadata to keep all columns
        final_df = pd.merge(df, results_df[['image_name', 'predicted_skin_type']], 
                           on='image_name', how='inner')
        
        # Save to CSV
        output_path = os.path.join(output_folder, 'isic2020_with_monk_skin_types.csv')
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Visualize results
        visualize_results(final_df, image_folder, output_folder)
        
        return final_df
    else:
        print("No results to process after feature extraction.")
        return None
    
def process_dataset_with_model(csv_path, image_folder, output_folder, model, device):
    """
    Process the ISIC dataset and classify images using a pre-trained model.
    
    Args:
        csv_path: Path to the ISIC 2020 metadata CSV
        image_folder: Path to the folder containing the images
        output_folder: Folder to save results
        model: Pre-trained skin tone classification model
        device: Compute device (CPU/GPU)
    
    Returns:
        DataFrame with added skin type predictions
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the metadata
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata with {len(df)} entries")
    
    # Setup image transforms
    transform = get_transforms()
    
    # Create lists to store results
    results = []
    
    # Process each image
    print("Processing images with pre-trained model...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['image_name']
        
        # Try to find the image file
        image_path = os.path.join(image_folder, f"{image_name}.jpg")
        
        # Try different extensions if not found
        if not os.path.exists(image_path):
            for ext in ['.png', '.jpeg']:
                alt_path = os.path.join(image_folder, f"{image_name}{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
            
            # Skip if image still not found
            if not os.path.exists(image_path):
                print(f"Image not found: {image_name}")
                continue
        
        # Predict skin type using the model
        skin_type = predict_with_pretrained_model(model, device, image_path, transform)
        
        if skin_type is not None:
            results.append({
                'image_name': image_name,
                'predicted_skin_type': skin_type
            })
    
    # Convert results to DataFrame
    if not results:
        print("No images could be processed with the model.")
        return None
    
    results_df = pd.DataFrame(results)
    print(f"Successfully processed {len(results_df)} images")
    
    # Merge with original metadata to keep all columns
    final_df = pd.merge(df, results_df, on='image_name', how='inner')
    
    # Save to CSV
    output_path = os.path.join(output_folder, 'isic2020_with_monk_skin_types.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Visualize results
    visualize_results(final_df, image_folder, output_folder)
    
    return final_df


def visualize_results(results_df, image_folder, output_folder):
    """
    Visualize the skin type classification results.
    
    Args:
        results_df: DataFrame with skin type predictions
        image_folder: Folder containing the images
        output_folder: Folder to save visualizations
    """
    # Plot distribution of predicted skin types
    plt.figure(figsize=(12, 6))
    results_df['predicted_skin_type'].value_counts().sort_index().plot(kind='bar', 
                                                                     color='skyblue')
    plt.title('Distribution of Predicted Monk Skin Types')
    plt.xlabel('Monk Skin Type (1-10)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(output_folder, 'monk_skin_type_distribution.png'), dpi=300)
    plt.close()
    
    # Create sample grid of images for each skin type
    for skin_type in range(1, 11):
        # Get sample images for this skin type
        type_df = results_df[results_df['predicted_skin_type'] == skin_type]
        
        # Take up to 9 samples
        samples = type_df.sample(min(9, len(type_df))) if len(type_df) > 0 else type_df
        
        if len(samples) > 0:
            # Determine grid size based on number of samples
            grid_size = min(3, len(samples))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            
            # Handle case of single image
            if grid_size == 1:
                axes = np.array([[axes]])
            
            axes = axes.flatten()
            
            for i, (_, row) in enumerate(samples.iterrows()):
                if i >= len(axes):
                    break
                    
                image_id = row['image_name'] if 'image_name' in row else row['image_id']
                image_path = os.path.join(image_folder, f"{image_id}.jpg")
                
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                    axes[i].set_title(f"ID: {image_id}")
                    axes[i].axis('off')
                
            # Hide any unused axes
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
                
            plt.suptitle(f'Sample Images for Monk Skin Type {skin_type}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'monk_type_{skin_type}_samples.png'), dpi=300)
            plt.close()

def load_or_train_model(training_data_path=None):
    """
    Load a pre-trained model or train a new one if training data is available.
    
    Args:
        training_data_path: Path to training data with labeled skin types
        
    Returns:
        Trained model and device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model architecture
    model = SkinToneClassifier(num_classes=10)
    model.to(device)
    
    # If we have training data, train the model
    if training_data_path and os.path.exists(training_data_path):
        # Load training data
        train_df = pd.read_csv(training_data_path)
        
        if 'image_path' in train_df.columns and 'monk_skin_type' in train_df.columns:
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
            
            # Split data
            train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
            
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
            
            # Train
            num_epochs = 20
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                running_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                
                train_loss = running_loss / len(train_loader.dataset)
                
                # Validation phase
                model.eval()
                running_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        running_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_loss = running_loss / len(val_loader.dataset)
                val_acc = correct / total
                
                print(f'Epoch {epoch+1}/{num_epochs} | '
                      f'Train Loss: {train_loss:.4f} | '
                      f'Val Loss: {val_loss:.4f} | '
                      f'Val Acc: {val_acc:.4f}')
                
                # Update scheduler
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'best_skin_tone_model.pth')
            
            # Load best model for inference
            model.load_state_dict(torch.load('best_skin_tone_model.pth'))
            print("Model training complete!")
        
        else:
            print("Training data doesn't have required columns (image_path, monk_skin_type)")
            model = None
    
    else:
        print("No training data provided. Cannot train model.")
        model = None
    
    return model, device

def main():
    """Main function to run the script."""

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_folder = os.path.join(data_dir, 'skin_type_analysis')
    model_folder = os.path.join(project_root, 'trained_model')
    
    # Path to pre-trained model if available
    pretrained_model_path = os.path.join(model_folder, 'monk_skin_tone_model.pth')
    if not os.path.exists(pretrained_model_path):
        pretrained_model_path = None  # Set to None if model doesn't exist
        print("No pre-trained model found at:", pretrained_model_path)
    else:
        print("Found pre-trained model at:", pretrained_model_path)
    
    # Path to training data with labeled Monk skin types (if available)
    training_data_path = os.path.join(data_dir, 'labeled_skin_types.csv')
    if not os.path.exists(training_data_path):
        training_data_path = None  # Set to None if labeled data doesn't exist
        print("No labeled training data found at:", training_data_path)
    else:
        print("Found labeled training data at:", training_data_path)
    
    # If we have a pre-trained model, use it
    if pretrained_model_path:
        print("Loading pre-trained model...")
        model, device = load_pretrained_model(pretrained_model_path)
        
        # Proceed with classification using the model
        # Need to modify process_dataset to work with the model
        results_df = process_dataset_with_model(csv_path, image_folder, output_folder, model, device)
        
    # If we have training data but no model, train a new model
    elif training_data_path:
        print("Training new model from labeled data...")
        model, device = load_or_train_model(training_data_path)
        
        if model:
            # Create model folder if it doesn't exist
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
                
            # Save the model
            model_save_path = os.path.join(model_folder, 'monk_skin_tone_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
            
            # Use the trained model for classification
            results_df = process_dataset_with_model(csv_path, image_folder, output_folder, model, device)
        else:
            # Fall back to clustering if model training failed
            print("Model training failed, falling back to clustering approach...")
            results_df = process_dataset(csv_path, image_folder, output_folder, max_images=None)
    
    # If we have neither model nor labeled data, use clustering
    else:
        print("Using clustering approach for classification...")
        # Process dataset using the clustering method
        results_df = process_dataset(csv_path, image_folder, output_folder, max_images=10000)
    
    # Print summary
    if results_df is not None:
        print("\nClassification complete!")
        print(f"Processed {len(results_df)} images.")
        print(f"Results saved to {os.path.join(output_folder, 'ISIC_2020_with_monk_skin_types.csv')}")
        
        print("\nMonk Skin Type Distribution:")
        print(results_df['predicted_skin_type'].value_counts().sort_index())
    else:
        print("Classification failed - no results generated.")

if __name__ == "__main__":
    main()