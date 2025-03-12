import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    print("Running K-means clustering...")
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
    
    plt.title('Distribution of Predicted Monk Skin Types')
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
        samples = type_df.sample(min(9, len(type_df)))
        
        # Determine grid size based on number of samples
        grid_size = min(3, int(np.ceil(np.sqrt(len(samples)))))
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
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_folder = os.path.join(data_dir, 'skin_type_analysis')
    
    # Process the dataset using clustering
    print("Using clustering approach for classification...")
    max_images = None  # Set to a number for testing, None for all images
    results_df = process_dataset(csv_path, image_folder, output_folder, max_images)
    
    # Print summary
    if results_df is not None:
        print("\nClustering-based classification complete!")
        print(f"Processed {len(results_df)} images.")
        print(f"Results saved to {os.path.join(output_folder, 'isic2020_with_monk_skin_types.csv')}")
        
        print("\nMonk Skin Type Distribution:")
        print(results_df['predicted_skin_type'].value_counts().sort_index())
    else:
        print("Classification failed - no results generated.")

if __name__ == "__main__":
    main()