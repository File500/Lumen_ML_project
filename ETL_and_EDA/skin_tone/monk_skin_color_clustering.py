import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture

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
    Uses both cropping and masking to focus only on the surrounding skin,
    excluding the lesion.
    
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
        
        # Convert to different color spaces for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create a combined mask to exclude:
        # 1. Very dark areas (likely not skin)
        # 2. The lesion itself
        
        # First, create basic brightness mask
        brightness_mask = gray > 30  # Exclude very dark pixels
        
        # Now create a lesion mask using multiple techniques
        
        # Method 1: Use saturation to identify lesions (often more saturated than skin)
        sat_channel = hsv[:,:,1]
        _, sat_mask = cv2.threshold(sat_channel, 180, 255, cv2.THRESH_BINARY)
        sat_mask = sat_mask > 0
        
        # Method 2: Use color deviation from mean skin tone
        # Compute mean color of the image (likely to be skin color)
        mean_color = np.mean(img, axis=(0, 1))
        
        # Create a color distance image
        color_dist = np.zeros_like(gray, dtype=np.float32)
        for i in range(3):  # For each BGR channel
            diff = img[:,:,i].astype(np.float32) - mean_color[i]
            color_dist += diff * diff
        color_dist = np.sqrt(color_dist)
        
        # Threshold to find areas with significant color deviation
        _, color_mask = cv2.threshold(color_dist, 100, 255, cv2.THRESH_BINARY)
        color_mask = color_mask > 0
        
        # Method 3: Focus on the border region of the image
        # Create a border mask (25% border around the image)
        h, w = img.shape[:2]
        border_width = int(min(h, w) * 0.25)
        border_mask = np.ones_like(gray, dtype=bool)
        
        # Remove the center part
        center_h_start = max(0, int(h/2 - border_width/2))
        center_h_end = min(h, int(h/2 + border_width/2))
        center_w_start = max(0, int(w/2 - border_width/2))
        center_w_end = min(w, int(w/2 + border_width/2))
        
        border_mask[center_h_start:center_h_end, center_w_start:center_w_end] = False
        
        # Combine masks:
        # 1. Keep bright areas (not dark)
        # 2. Exclude saturated areas (likely lesion)
        # 3. Exclude areas with different color (likely lesion)
        # 4. Focus on border regions more likely to be normal skin
        
        # Start with brightness mask
        combined_mask = brightness_mask.copy()
        
        # Exclude saturated areas and color-different areas
        combined_mask = combined_mask & ~sat_mask & ~color_mask
        
        # Enhance weight of border regions
        combined_mask = combined_mask | (border_mask & brightness_mask)
        
        # Check if the mask is too restrictive
        if np.sum(combined_mask) < (np.prod(img.shape[:2]) * 0.05):  
            # If less than 5% of pixels remain, fall back to just brightness mask
            combined_mask = brightness_mask
            print(f"Warning: Mask too restrictive for {image_path}, using only brightness mask")
        
        # Convert to LAB color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Extract L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        # Apply mask to color channels
        l_masked = l_channel[combined_mask]
        a_masked = a_channel[combined_mask]
        b_masked = b_channel[combined_mask]
        
        # Handle case where mask is too restrictive
        if len(l_masked) < 100:  # Ensure we have enough pixels
            print(f"Warning: Not enough pixels after masking for {image_path}")
            l_masked = l_channel.flatten()
            a_masked = a_channel.flatten()
            b_masked = b_channel.flatten()
        
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
    
def save_masked_image_for_debug(image_path, output_folder, method_name=None, max_samples=10):
    """
    Save visualizations of the masked image to see what areas are being used
    for skin tone analysis. Useful for debugging the lesion exclusion.
    
    Args:
        image_path: Path to the original image
        output_folder: Where to save the visualization
        method_name: Name of clustering method (or None for general)
        max_samples: Maximum number of debug images to create per method
    """
    # Use a counter to limit the number of debug images created
    if not hasattr(save_masked_image_for_debug, 'counters'):
        save_masked_image_for_debug.counters = {}
    
    # Initialize counter for this method if needed
    if method_name not in save_masked_image_for_debug.counters:
        save_masked_image_for_debug.counters[method_name] = 0
        
    # Check if we've reached the max samples for this method
    if save_masked_image_for_debug.counters[method_name] >= max_samples:
        return
        
    try:
        # Create debug output folder
        if method_name:
            debug_folder = os.path.join(output_folder, method_name, 'mask_debug')
        else:
            debug_folder = os.path.join(output_folder, 'mask_debug')
            
        os.makedirs(debug_folder, exist_ok=True)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return
            
        # Crop black lines
        img = crop_black_lines(img)
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create masks
        brightness_mask = gray > 30
        
        sat_channel = hsv[:,:,1]
        _, sat_mask = cv2.threshold(sat_channel, 180, 255, cv2.THRESH_BINARY)
        sat_mask = sat_mask > 0
        
        mean_color = np.mean(img, axis=(0, 1))
        color_dist = np.zeros_like(gray, dtype=np.float32)
        for i in range(3):
            diff = img[:,:,i].astype(np.float32) - mean_color[i]
            color_dist += diff * diff
        color_dist = np.sqrt(color_dist)
        
        _, color_mask = cv2.threshold(color_dist, 100, 255, cv2.THRESH_BINARY)
        color_mask = color_mask > 0
        
        h, w = img.shape[:2]
        border_width = int(min(h, w) * 0.25)
        border_mask = np.ones_like(gray, dtype=bool)
        center_h_start = max(0, int(h/2 - border_width/2))
        center_h_end = min(h, int(h/2 + border_width/2))
        center_w_start = max(0, int(w/2 - border_width/2))
        center_w_end = min(w, int(w/2 + border_width/2))
        border_mask[center_h_start:center_h_end, center_w_start:center_w_end] = False
        
        # Combined mask
        combined_mask = brightness_mask.copy()
        combined_mask = combined_mask & ~sat_mask & ~color_mask
        combined_mask = combined_mask | (border_mask & brightness_mask)
        
        # Create visualization images
        brightness_viz = img.copy()
        brightness_viz[~brightness_mask] = [0, 0, 0]
        
        sat_viz = img.copy()
        sat_viz[sat_mask] = [0, 0, 255]  # Red overlay for saturated areas
        
        color_viz = img.copy()
        color_viz[color_mask] = [255, 0, 0]  # Blue overlay for color-different areas
        
        border_viz = img.copy()
        border_viz[~border_mask] = [0, 0, 0]
        
        combined_viz = img.copy()
        combined_viz[~combined_mask] = [0, 0, 0]
        
        # Extract image name for the output filename
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save visualizations
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_original.jpg"), img)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_1_brightness.jpg"), brightness_viz)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_2_saturation.jpg"), sat_viz)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_3_color.jpg"), color_viz)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_4_border.jpg"), border_viz)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_5_combined.jpg"), combined_viz)
        
        # Also save the LAB color space visualization
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        # Normalize LAB channels for visualization
        l_norm = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        a_norm = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX)
        b_norm = cv2.normalize(b_channel, None, 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_6_L_channel.jpg"), l_norm)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_7_A_channel.jpg"), a_norm)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_8_B_channel.jpg"), b_norm)
        
        # Increment counter for this method
        save_masked_image_for_debug.counters[method_name] += 1
        
    except Exception as e:
        print(f"Error creating debug images for {image_path}: {e}")

def classify_with_kmeans(features_df, output_folder):
    """
    Classify skin types using K-means clustering.
    
    Args:
        features_df: DataFrame with extracted skin features
        output_folder: Folder to save results
        
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
    skin_tone_scores = []
    for center in cluster_centers:
        # Invert L value so higher value = darker skin
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]         # a can be negative (green) or positive (red)
        b_value = center[b_index]         # b can be negative (blue) or positive (yellow)
        
        # Simple weighted score
        score = (0.55 * l_value) + (0.25 * a_value) + (0.20 * b_value)
        skin_tone_scores.append(score)
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    sorted_indices = np.argsort(skin_tone_scores)
    
    # Map original clusters to Monk skin types (1-10)
    cluster_to_skin_type = {}
    for skin_type, cluster in enumerate(sorted_indices, 1):
        cluster_to_skin_type[cluster] = skin_type
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in clusters]
    
    return results_df

def classify_with_gmm(features_df, output_folder):
    """
    Classify skin types using Gaussian Mixture Model clustering.
    
    Args:
        features_df: DataFrame with extracted skin features
        output_folder: Folder to save results
        
    Returns:
        DataFrame with added skin type predictions
    """
    # Features to use for clustering
    feature_cols = [col for col in features_df.columns if col.startswith(('avg_', 'std_', 'dom'))]
    
    # Create feature matrix
    X = features_df[feature_cols].copy()
    
    # Normalize features
    X = (X - X.mean()) / X.std()
    
    # Use GMM to cluster the data into 10 clusters
    print("Running GMM clustering...")
    gmm = GaussianMixture(n_components=10, random_state=42, covariance_type='full', n_init=3)
    gmm.fit(X)
    
    # Get cluster assignments
    clusters = gmm.predict(X)
    
    # Get cluster centers (means of each Gaussian component)
    cluster_centers = gmm.means_
    
    # Find average L value (lightness) for each cluster
    l_index = feature_cols.index('avg_l')
    a_index = feature_cols.index('avg_a')
    b_index = feature_cols.index('avg_b')
    
    # Calculate a combined skin tone score using L, a, b values
    skin_tone_scores = []
    for center in cluster_centers:
        # Invert L value so higher value = darker skin
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]         # a can be negative (green) or positive (red)
        b_value = center[b_index]         # b can be negative (blue) or positive (yellow)
        
        # Simple weighted score
        score = (0.55 * l_value) + (0.25 * a_value) + (0.20 * b_value)
        skin_tone_scores.append(score)
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    sorted_indices = np.argsort(skin_tone_scores)
    
    # Map original clusters to Monk skin types (1-10)
    cluster_to_skin_type = {}
    for skin_type, cluster in enumerate(sorted_indices, 1):
        cluster_to_skin_type[cluster] = skin_type
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in clusters]
    
    return results_df

def classify_with_spectral(features_df, output_folder):
    """
    Classify skin types using Spectral Clustering.
    
    Args:
        features_df: DataFrame with extracted skin features
        output_folder: Folder to save results
        
    Returns:
        DataFrame with added skin type predictions
    """
    # Features to use for clustering
    feature_cols = [col for col in features_df.columns if col.startswith(('avg_', 'std_', 'dom'))]
    
    # Create feature matrix
    X = features_df[feature_cols].copy()
    
    # Normalize features
    X = (X - X.mean()) / X.std()
    
    # Use Spectral Clustering
    print("Running Spectral Clustering...")
    spectral = SpectralClustering(n_clusters=10, random_state=42, affinity='nearest_neighbors')
    clusters = spectral.fit_predict(X)
    
    # For Spectral Clustering, we need to compute cluster centers manually
    cluster_centers = []
    for i in range(10):
        # Get indices of points in this cluster
        indices = np.where(clusters == i)[0]
        
        # If cluster is empty, use zeros
        if len(indices) == 0:
            center = np.zeros(len(feature_cols))
        else:
            # Compute mean of points in this cluster
            center = X.iloc[indices].mean().values
            
        cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers)
    
    # Find average L value (lightness) for each cluster
    l_index = feature_cols.index('avg_l')
    a_index = feature_cols.index('avg_a')
    b_index = feature_cols.index('avg_b')
    
    # Calculate a combined skin tone score using L, a, b values
    skin_tone_scores = []
    for center in cluster_centers:
        # Invert L value so higher value = darker skin
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]         # a can be negative (green) or positive (red)
        b_value = center[b_index]         # b can be negative (blue) or positive (yellow)
        
        # Simple weighted score
        score = (0.55 * l_value) + (0.25 * a_value) + (0.20 * b_value)
        skin_tone_scores.append(score)
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    sorted_indices = np.argsort(skin_tone_scores)
    
    # Map original clusters to Monk skin types (1-10)
    cluster_to_skin_type = {}
    for skin_type, cluster in enumerate(sorted_indices, 1):
        cluster_to_skin_type[cluster] = skin_type
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in clusters]
    
    return results_df

def classify_with_dbscan(features_df, output_folder):
    """
    Classify skin types using DBSCAN clustering.
    
    Args:
        features_df: DataFrame with extracted skin features
        output_folder: Folder to save results
        
    Returns:
        DataFrame with added skin type predictions
    """
    # Features to use for clustering
    feature_cols = [col for col in features_df.columns if col.startswith(('avg_', 'std_', 'dom'))]
    
    # Create feature matrix
    X = features_df[feature_cols].copy()
    
    # Normalize features
    X_norm = (X - X.mean()) / X.std()
    
    # Use DBSCAN clustering
    print("Running DBSCAN clustering...")
    # DBSCAN parameters need tuning for your specific dataset
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_norm)
    
    # Handle noise points (label -1)
    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    
    # If we have noise points or fewer than 10 clusters, we need to adjust
    if -1 in unique_clusters or n_clusters < 10:
        print(f"DBSCAN found {n_clusters} clusters including noise. Adjusting...")
        
        # Find the optimal number of clusters using KMeans on DBSCAN clusters
        # First, identify non-noise points
        if -1 in unique_clusters:
            non_noise_mask = clusters != -1
            non_noise_X = X_norm.iloc[non_noise_mask]
            non_noise_indices = np.where(non_noise_mask)[0]
            
            # Determine how many more clusters we need
            remaining_clusters = 10 - (n_clusters - 1)  # Subtract 1 for noise cluster
            
            if remaining_clusters > 0 and len(non_noise_X) > 0:
                # Use KMeans to find more clusters in the non-noise data
                kmeans = KMeans(n_clusters=min(remaining_clusters, len(non_noise_X)), 
                               random_state=42, n_init=10)
                sub_clusters = kmeans.fit_predict(non_noise_X)
                
                # Assign new cluster IDs (starting from max existing cluster + 1)
                max_cluster = max(clusters[clusters != -1]) if any(clusters != -1) else 0
                for i, idx in enumerate(non_noise_indices):
                    clusters[idx] = max_cluster + 1 + sub_clusters[i]
        
        # If we still don't have enough clusters, use KMeans for the whole dataset
        if len(np.unique(clusters)) < 10:
            print("DBSCAN produced too few clusters, falling back to KMeans")
            kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_norm)
            cluster_centers = kmeans.cluster_centers_
        else:
            # Compute cluster centers for DBSCAN manually
            cluster_centers = []
            for i in np.unique(clusters):
                if i == -1:  # Skip noise points for center calculation
                    continue
                    
                indices = np.where(clusters == i)[0]
                center = X.iloc[indices].mean().values
                cluster_centers.append(center)
            
            # If we have fewer than 10 centers, pad with zeros
            while len(cluster_centers) < 10:
                center = np.zeros(len(feature_cols))
                cluster_centers.append(center)
                
            cluster_centers = np.array(cluster_centers)
    else:
        # Compute cluster centers for each non-noise cluster
        cluster_centers = []
        for i in range(10):  # Assuming clusters are labeled 0-9
            indices = np.where(clusters == i)[0]
            
            if len(indices) == 0:
                center = np.zeros(len(feature_cols))
            else:
                center = X.iloc[indices].mean().values
                
            cluster_centers.append(center)
            
        cluster_centers = np.array(cluster_centers)
    
    # Find average L value (lightness) for each cluster
    l_index = feature_cols.index('avg_l')
    a_index = feature_cols.index('avg_a')
    b_index = feature_cols.index('avg_b')
    
    # Calculate a combined skin tone score using L, a, b values
    skin_tone_scores = []
    for center in cluster_centers:
        # Invert L value so higher value = darker skin
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]         # a can be negative (green) or positive (red)
        b_value = center[b_index]         # b can be negative (blue) or positive (yellow)
        
        # Simple weighted score
        score = (0.55 * l_value) + (0.25 * a_value) + (0.20 * b_value)
        skin_tone_scores.append(score)
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    sorted_indices = np.argsort(skin_tone_scores)
    
    # Map original clusters to Monk skin types (1-10)
    cluster_to_skin_type = {}
    valid_clusters = [c for c in np.unique(clusters) if c != -1]
    
    # Map existing clusters to skin types
    for skin_type, i in enumerate(sorted_indices[:len(valid_clusters)], 1):
        if i < len(valid_clusters):
            cluster_id = valid_clusters[i]
            cluster_to_skin_type[cluster_id] = skin_type
    
    # Handle any unmapped clusters
    for c in valid_clusters:
        if c not in cluster_to_skin_type:
            # Assign to closest skin type based on center
            if len(cluster_centers) > c:
                center = cluster_centers[c]
                l_value = 100 - center[l_index]
                score = (0.55 * l_value) + (0.25 * center[a_index]) + (0.20 * center[b_index])
                
                # Find closest skin type score
                closest_skin_type = min(range(1, 11), 
                                       key=lambda st: abs(skin_tone_scores[sorted_indices[st-1]] - score))
                cluster_to_skin_type[c] = closest_skin_type
            else:
                # Fallback
                cluster_to_skin_type[c] = 1
    
    # Handle noise points (-1) - assign them to skin type 1 (lightest)
    if -1 in np.unique(clusters):
        cluster_to_skin_type[-1] = 1
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type.get(c, 1) for c in clusters]
    
    return results_df

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
        samples = type_df.sample(min(80, len(type_df)))
        
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

def process_with_multiple_methods(csv_path, image_folder, output_folder, features_df=None, max_images=None, save_debug_masks=True):
    """
    Process the dataset using multiple clustering methods.
    
    Args:
        csv_path: Path to CSV file
        image_folder: Path to image folder
        output_folder: Base output folder
        features_df: Pre-extracted features (if None, will extract features)
        max_images: Maximum number of images to process
        save_debug_masks: Whether to save mask visualizations
    """
    # Load metadata
    df = pd.read_csv(csv_path)
    print(f"Loaded metadata with {len(df)} entries")
    
    # Limit to max_images if specified
    if max_images is not None:
        df = df.head(max_images)
        print(f"Using first {len(df)} images")
    
    # If features not provided, extract them
    if features_df is None:
        # Create a list to store results
        results = []
        
        # Process each image
        print("Extracting features from images...")
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
            
            # Save general debug masks
            if save_debug_masks:
                save_masked_image_for_debug(image_path, output_folder, method_name='general')
            
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
        
        features_df = pd.DataFrame(results)
        print(f"Successfully extracted features from {len(features_df)} images")
        
        # Save features for reuse
        features_path = os.path.join(output_folder, 'extracted_features.csv')
        features_df.to_csv(features_path, index=False)
        print(f"Saved extracted features to {features_path}")
    
    # Run each clustering method
    clustering_methods = {
        'kmeans': classify_with_kmeans,
        'gmm': classify_with_gmm,
        'spectral': classify_with_spectral,
        'dbscan': classify_with_dbscan
    }
    
    results = {}
    
    for method_name, method_func in clustering_methods.items():
        print(f"\nRunning {method_name.upper()} clustering...")
        
        # Create method-specific output folder
        method_folder = os.path.join(output_folder, method_name)
        os.makedirs(method_folder, exist_ok=True)
        
        # Run clustering method
        method_results = method_func(features_df, method_folder)
        
        # Merge with original metadata
        final_df = pd.merge(df, method_results[['image_name', 'predicted_skin_type']], 
                           on='image_name', how='inner')
        
        # Save to CSV
        output_path = os.path.join(method_folder, 'ISIC_2020_with_monk_skin_types.csv')
        final_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Save debug masks for this method
        if save_debug_masks:
            # Get a few random samples for each skin type
            for skin_type in range(1, 11):
                type_samples = final_df[final_df['predicted_skin_type'] == skin_type]
                if len(type_samples) > 0:
                    sample_count = min(2, len(type_samples))  # 2 samples per skin type
                    samples = type_samples.sample(sample_count)
                    
                    for _, sample in samples.iterrows():
                        image_name = sample['image_name']
                        image_path = os.path.join(image_folder, f"{image_name}.jpg")
                        
                        # Try different extensions
                        if not os.path.exists(image_path):
                            for ext in ['.png', '.jpeg']:
                                alt_path = os.path.join(image_folder, f"{image_name}{ext}")
                                if os.path.exists(alt_path):
                                    image_path = alt_path
                                    break
                        
                        if os.path.exists(image_path):
                            save_masked_image_for_debug(image_path, output_folder, method_name=method_name)
        
        # Visualize results
        visualize_results(final_df, image_folder, method_folder)
        
        # Store results for comparison
        results[method_name] = final_df
        
        print(f"\n{method_name.upper()} Skin Type Distribution:")
        print(final_df['predicted_skin_type'].value_counts().sort_index())
    
    # Create comparison visualization
    compare_methods(results, output_folder)
    
    return results

def compare_methods(results_dict, output_folder):
    """
    Create a visualization comparing the distributions from different methods.
    
    Args:
        results_dict: Dictionary of DataFrames with results from different methods
        output_folder: Folder to save the comparison
    """
    plt.figure(figsize=(15, 8))
    
    # Get all methods
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    # Set up color cycle
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum']
    
    # Create subplots
    for i, method in enumerate(methods):
        plt.subplot(2, (n_methods + 1) // 2, i + 1)
        
        # Get distribution
        dist = results_dict[method]['predicted_skin_type'].value_counts().sort_index()
        
        # Plot
        dist.plot(kind='bar', color=colors[i % len(colors)])
        plt.title(f'{method.upper()} Distribution')
        plt.xlabel('Skin Type')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add count labels
        for j, count in enumerate(dist):
            plt.text(j, count + 0.5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'method_comparison.png'), dpi=300)
    plt.close()

def main():
    """Main function to run the script."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_folder = os.path.join(data_dir, 'skin_type_analysis', 'clustering_comparison')
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Set number of images to process
    max_images = 100  # Set to None for all images
    
    # Process with multiple clustering methods
    results = process_with_multiple_methods(csv_path, image_folder, output_folder, 
                                           features_df=None, max_images=max_images,
                                           save_debug_masks=True)
    
    if results:
        print("\nClustering comparison complete!")
    else:
        print("Clustering comparison failed.")

if __name__ == "__main__":
    main()