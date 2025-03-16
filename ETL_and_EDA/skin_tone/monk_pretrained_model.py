import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.skin_tone_model import SkinToneClassifier, ResNetSkinToneClassifier, EfficientNetSkinToneClassifier

# Define image transformations for the pre-trained model
def get_transforms():
    return transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                    std=[0.229, 0.224, 0.225])
    ])


def load_pretrained_model(model_path):
    """
    Load a pre-trained skin tone classification model.
    
    Args:
        model_path: Path to pre-trained model
    
    Returns:
        Loaded model and compute device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = SkinToneClassifier(num_classes=10)
    model = EfficientNetSkinToneClassifier(num_classes=10)
    
    # Load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded MobileNetV2 model from {model_path}")
    
    model.to(device)
    model.eval()
    
    return model, device

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
    print("Processing images with pre-trained MobileNetV2 model...")
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
    output_path = os.path.join(output_folder, 'ISIC_2020_model_output_with_monk_skin_types.csv')
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
    skin_type_counts = results_df['predicted_skin_type'].value_counts().sort_index()
    skin_type_counts.plot(kind='bar', color='skyblue')
    
    plt.title('Distribution of Predicted Monk Skin Types (MobileNetV2)')
    plt.xlabel('Monk Skin Type (1-10)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for i, count in enumerate(skin_type_counts):
        plt.text(i, count + 0.5, str(count), ha='center')
    
    plt.savefig(os.path.join(output_folder, 'monk_skin_type_distribution.png'), dpi=300)
    plt.close()
    
    # Create sample grid of images for each skin type
    for skin_type in range(1, 11):
        # Get sample images for this skin type
        type_df = results_df[results_df['predicted_skin_type'] == skin_type]
        
        # Skip if no images have this skin type
        if len(type_df) == 0:
            continue
            
        # Take up to 9 samples
        samples = type_df.sample(min(50, len(type_df)))
        
        # Determine grid size based on number of samples
        grid_size = min(10, int(np.ceil(np.sqrt(len(samples)))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        
        # Handle case of single image
        if grid_size == 1:
            axes = np.array([[axes]])
        
        axes = axes.flatten()
        
        for i, (_, row) in enumerate(samples.iterrows()):
            if i >= len(axes):
                break
                
            image_id = row['image_name']
            
            # Try different extensions
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = os.path.join(image_folder, f"{image_id}{ext}")
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            
            if image_path:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img)
                    axes[i].set_title(f"ID: {image_id}")
                    axes[i].axis('off')
            
        # Hide any unused axes
        for j in range(len(samples), len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(f'Sample Images for Monk Skin Type {skin_type}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'monk_type_{skin_type}_samples.png'), dpi=300)
        plt.close()

def main():
    """Main function to run the script."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_folder = os.path.join(data_dir, 'skin_type_analysis', 'model_predictions')
    model_folder = os.path.join(project_root, 'trained_model')

    os.makedirs(output_folder, exist_ok=True)
    
    # Path to pre-trained model
    model_path = os.path.join(model_folder, 'best_monk_skin_tone_model.pth')
    
    # Load pre-trained model
    model, device = load_pretrained_model(model_path)
    
    # Process dataset using the model
    results_df = process_dataset_with_model(csv_path, image_folder, output_folder, model, device)
    
    # Print summary
    if results_df is not None:
        print("\nMobileNetV2 model-based classification complete!")
        print(f"Processed {len(results_df)} images.")
        print(f"Results saved to {os.path.join(output_folder, 'ISIC_2020_model_output_with_monk_skin_types.csv')}")
        
        print("\nMonk Skin Type Distribution:")
        print(results_df['predicted_skin_type'].value_counts().sort_index())
    else:
        print("Classification failed - no results generated.")

if __name__ == "__main__":
    main()