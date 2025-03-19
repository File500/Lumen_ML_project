import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    Extract features related to skin tone with corrected masking.
    Properly excludes lesions and keeps normal skin.
    
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
        
        # Convert to color spaces
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # IMPORTANT: For all masks below, True (1) means we KEEP that pixel,
        # False (0) means we EXCLUDE that pixel
        
        # Approach: Lesions are typically darker, more red, and more saturated than surrounding skin
        
        # 1. Lightness mask (L channel)
        # KEEP pixels that are LIGHTER (higher L values)
        l_mean = np.mean(l_channel)
        l_std = np.std(l_channel)
        #l_mask = l_channel > l_mean - 0.5 * l_std  # KEEP pixels that are lighter
        l_mask = l_channel > l_mean
        
        # 2. Redness mask (A channel)
        # KEEP pixels that are LESS RED (lower A values)
        a_mean = np.mean(a_channel)
        a_std = np.std(a_channel)
        #a_mask = a_channel < a_mean + 0.5 * a_std  # KEEP pixels that are less red
        a_mask = a_channel < a_mean
        
        # 3. Saturation mask (S channel)
        # KEEP pixels that are LESS SATURATED (lower S values)
        s_mean = np.mean(s_channel)
        s_std = np.std(s_channel)
        s_mask = s_channel < s_mean # KEEP pixels that are less saturated
        
        # 4. Border mask - assume the edges are more likely to be normal skin
        h, w = img.shape[:2]
        border_width = int(min(h, w) * 0.2)  # 20% border width
        
        # Create a border mask where True = border region (to keep)
        border_mask = np.zeros((h, w), dtype=bool)
        
        # Top, bottom, left, right borders
        border_mask[:border_width, :] = True  # Top border
        border_mask[-border_width:, :] = True  # Bottom border
        border_mask[:, :border_width] = True  # Left border
        border_mask[:, -border_width:] = True  # Right border

        center_x = w // 2
        half_strip_width = int(w * 0.5 / 2)
        
        # Create mask (True = keep, False = exclude)
        vertical_mask = np.ones((h, w), dtype=bool)
        
        # Set central vertical strip to False (exclude)
        x_min = max(0, center_x - half_strip_width)
        x_max = min(w, center_x + half_strip_width)
        
        vertical_mask[:, x_min:x_max] = False

        edges = cv2.Canny(l_channel.astype(np.uint8), 50, 150)
        kernel = np.ones((7,7), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = edge_mask > 0
        
        # Combine the masks
        # 1. First approach: Keep pixels that are likely normal skin based on color features
        color_mask = l_mask & a_mask & s_mask  # Pixels that are light AND less red AND less saturated
        
        # 2. Second approach: Keep border regions
        # We will combine these approaches with OR because we want to keep EITHER
        # pixels that look like normal skin OR pixels that are in the border
        #final_mask = color_mask | border_mask
        final_mask = color_mask  & ~edge_mask
        
        # Apply the mask to extract features from normal skin areas
        l_masked = l_channel[final_mask]
        a_masked = a_channel[final_mask]
        b_masked = b_channel[final_mask]
        
        # Check if we have enough pixels
        if len(l_masked) < (l_channel.size * 0.1):  # Less than 10% of the image
            # If mask is too restrictive, fall back to just the L mask (lightness)
            # which is usually the most reliable for distinguishing skin from lesion
            l_masked = l_channel[l_mask]
            a_masked = a_channel[l_mask]
            b_masked = b_channel[l_mask]
            
            print(f"Warning: Final mask too restrictive for {image_path}, using L mask")
            
            # If still too few pixels, use the entire image
            if len(l_masked) < 100:
                l_masked = l_channel.flatten()
                a_masked = a_channel.flatten()
                b_masked = b_channel.flatten()
                print(f"Warning: All masks too restrictive for {image_path}, using full image")

        # After applying your mask but before the KMeans step:
        if len(l_masked) > 200:
            # Take only the top 50% brightest pixels
            brightness_threshold = np.percentile(l_masked, 50)
            brightest_indices = l_masked >= brightness_threshold
            l_masked = l_masked[brightest_indices]
            a_masked = a_masked[brightest_indices]
            b_masked = b_masked[brightest_indices]
        
        # Use KMeans to find dominant colors
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
        features['skin_lesion_contrast'] = np.mean(l_masked) - np.percentile(l_channel, 5)
        features['red_ratio'] = features['avg_a'] / features['avg_l']
        
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
    Save visualizations of the improved masking approach.
    
    Args:
        image_path: Path to the original image
        output_folder: Where to save the visualization
        method_name: Name of clustering method (or None for general)
        max_samples: Maximum number of debug images to create per method
    """
    # Counter code remains the same
    if not hasattr(save_masked_image_for_debug, 'counters'):
        save_masked_image_for_debug.counters = {}
    
    if method_name not in save_masked_image_for_debug.counters:
        save_masked_image_for_debug.counters[method_name] = 0
        
    if save_masked_image_for_debug.counters[method_name] >= max_samples:
        return
        
    try:
        # Create debug folder - no changes needed
        if method_name:
            debug_folder = os.path.join(output_folder, method_name, 'mask_debug')
        else:
            debug_folder = os.path.join(output_folder, 'mask_debug')
            
        os.makedirs(debug_folder, exist_ok=True)
        
        # Read image - no changes needed
        img = cv2.imread(image_path)
        if img is None:
            return
            
        # Crop black lines - no changes needed
        img = crop_black_lines(img)
        
        # Color conversions - no changes needed
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # Create masks with the improved thresholds
        # 1. More aggressive lightness mask
        l_mean = np.mean(l_channel)
        l_std = np.std(l_channel)
        l_mask = l_channel > l_mean  # More aggressive threshold
        
        # 2. Stricter redness mask
        a_mean = np.mean(a_channel)
        a_std = np.std(a_channel)
        a_mask = a_channel < a_mean# Stricter threshold
        
        # 3. Stricter saturation mask
        s_mean = np.mean(s_channel)
        s_std = np.std(s_channel)
        s_mask = s_channel < s_mean # Stricter threshold
        
        # 4. Smaller border mask
        h, w = img.shape[:2]
        border_width = int(min(h, w) * 0.20)  
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[:border_width, :] = True
        border_mask[-border_width:, :] = True
        border_mask[:, :border_width] = True
        border_mask[:, -border_width:] = True

        # Calculate central region boundaries
        center_x = w // 2
        half_strip_width = int(w * 0.5 / 2)
        
        # Create mask (True = keep, False = exclude)
        vertical_mask = np.ones((h, w), dtype=bool)
        
        # Set central vertical strip to False (exclude)
        x_min = max(0, center_x - half_strip_width)
        x_max = min(w, center_x + half_strip_width)
        
        vertical_mask[:, x_min:x_max] = False
        
        # 5. NEW: Add edge detection
        edges = cv2.Canny(l_channel.astype(np.uint8), 50, 150)
        kernel = np.ones((7,7), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = edge_mask > 0
        
        # Combine masks
        color_mask = l_mask & a_mask & s_mask
        
        # Old mask for comparison
        old_final_mask = color_mask | border_mask
        
        # New mask that also excludes edges
        new_final_mask = color_mask & ~edge_mask
        
        # Create visualizations - similar to before but with some additions
        original = img.copy()
        
        # Channel visualizations - no changes needed
        l_viz = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        a_viz = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX)
        b_viz = cv2.normalize(b_channel, None, 0, 255, cv2.NORM_MINMAX)
        s_viz = cv2.normalize(s_channel, None, 0, 255, cv2.NORM_MINMAX)
        
        l_viz_color = cv2.cvtColor(l_viz, cv2.COLOR_GRAY2BGR)
        a_viz_color = cv2.cvtColor(a_viz, cv2.COLOR_GRAY2BGR)
        s_viz_color = cv2.cvtColor(s_viz, cv2.COLOR_GRAY2BGR)
        
        # Kept regions visualizations
        l_kept = img.copy()
        l_kept[~l_mask] = [0, 0, 0]
        
        a_kept = img.copy()
        a_kept[~a_mask] = [0, 0, 0]
        
        s_kept = img.copy()
        s_kept[~s_mask] = [0, 0, 0]
        
        border_kept = img.copy()
        border_kept[~border_mask] = [0, 0, 0]

        central_kept = img.copy()
        central_kept[~vertical_mask] = [0, 0, 0]
        
        color_kept = img.copy()
        color_kept[~color_mask] = [0, 0, 0]
        
        # NEW: Edge visualization
        edge_viz = np.zeros_like(img)
        edge_viz[edge_mask] = [0, 0, 255]  # Red for edges
        
        # Visualize old and new final masks
        old_final_kept = img.copy()
        old_final_kept[~old_final_mask] = [0, 0, 0]
        
        new_final_kept = img.copy()
        new_final_kept[~new_final_mask] = [0, 0, 0]
        
        # Exclusion visualizations
        l_excluded = img.copy()
        l_excluded[l_mask] = [0, 0, 0]
        
        a_excluded = img.copy()
        a_excluded[a_mask] = [0, 0, 0]
        
        s_excluded = img.copy()
        s_excluded[s_mask] = [0, 0, 0]
        
        color_excluded = img.copy()
        color_excluded[color_mask] = [0, 0, 0]
        
        # NEW: Edge exclusion visualization
        non_edge_excluded = img.copy()
        non_edge_excluded[~edge_mask] = [0, 0, 0]  # Show only edges
        
        old_final_excluded = img.copy()
        old_final_excluded[old_final_mask] = [0, 0, 0]
        
        new_final_excluded = img.copy()
        new_final_excluded[new_final_mask] = [0, 0, 0]
        
        # Extract image name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save visualizations - standard images
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_01_original.jpg"), original)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_02_L_channel.jpg"), l_viz_color)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_03_A_channel.jpg"), a_viz_color)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_04_S_channel.jpg"), s_viz_color)
        
        # Save mask visualizations
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_05_L_kept.jpg"), l_kept)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_06_A_kept.jpg"), a_kept)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_07_S_kept.jpg"), s_kept)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_08_border_kept.jpg"), border_kept)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_08a_central_kept.jpg"), central_kept)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_09_color_kept.jpg"), color_kept)
        
        # NEW: Save new visualizations
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_10a_edges_detected.jpg"), edge_viz)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_10b_old_final_kept.jpg"), old_final_kept)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_10c_new_final_kept.jpg"), new_final_kept)
        
        # Save exclusion visualizations
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_11_L_excluded.jpg"), l_excluded)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_12_A_excluded.jpg"), a_excluded)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_13_S_excluded.jpg"), s_excluded)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_14_color_excluded.jpg"), color_excluded)
        
        # NEW: Save new exclusion visualizations
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_15a_edges_only.jpg"), non_edge_excluded)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_15b_old_final_excluded.jpg"), old_final_excluded)
        cv2.imwrite(os.path.join(debug_folder, f"{img_name}_15c_new_final_excluded.jpg"), new_final_excluded)
        
        # Increment counter
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
    #X = (X - X.mean()) / X.std()
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Use KMeans to cluster the data into 10 clusters (for 10 Monk skin types)
    print("Running K-means clustering...")
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=20)
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
        score = (0.4 * l_value) + (0.30 * a_value) + (0.30 * b_value)
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
    Classify skin types using Gaussian Mixture Model clustering with improved differentiation.
    
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
    
    # Different preprocessing for GMM to differentiate from K-means
    # Apply non-linear transformation to enhance subtle differences
    gmm_X = X.copy()
    
    # Emphasize differences in a and b channels (color information)
    for col in feature_cols:
        if 'avg_a' in col or 'dom' in col and 'a' in col:
            # Apply non-linear transformation to a channel
            gmm_X[col] = np.sign(gmm_X[col]) * np.abs(gmm_X[col])**0.8
        if 'avg_b' in col or 'dom' in col and 'b' in col:
            # Apply non-linear transformation to b channel
            gmm_X[col] = np.sign(gmm_X[col]) * np.abs(gmm_X[col])**0.8
    
    # Normalize features
    gmm_X = (gmm_X - gmm_X.mean()) / gmm_X.std()
    
    # Try different GMM configurations
    print("Testing GMM configurations...")
    best_aic = float('inf')
    best_gmm = None
    best_config = None
    
    for cov_type in ['full', 'tied', 'diag']:
        for n_components in [10, 12, 15]:
            for reg_covar in [1e-4, 1e-3, 1e-2]:
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=42,
                    covariance_type=cov_type,
                    reg_covar=reg_covar,
                    n_init=5,
                    init_params='kmeans'  # Initialize using K-means but then diverge
                )
                
                try:
                    gmm.fit(gmm_X)
                    aic = gmm.aic(gmm_X)
                    
                    # Check cluster sizes
                    clusters = gmm.predict(gmm_X)
                    counts = np.bincount(clusters, minlength=n_components)
                    
                    # Calculate coefficient of variation (lower = more balanced)
                    cv = np.std(counts) / np.mean(counts)
                    
                    config = f"n={n_components}, cov={cov_type}, reg={reg_covar}"
                    print(f"  {config}: AIC={aic:.1f}, CV={cv:.3f}")
                    
                    # We want a balance between good fit (low AIC) and balance (low CV)
                    # Use a combined score
                    score = aic + cv * 1000  # Scale CV to be comparable with AIC
                    
                    if score < best_aic:
                        best_aic = score
                        best_gmm = gmm
                        best_config = config
                except Exception as e:
                    print(f"  Error with {cov_type}, n={n_components}, reg={reg_covar}: {e}")
    
    if best_gmm is None:
        print("All GMM configurations failed. Falling back to K-means.")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        kmeans.fit(gmm_X)
        clusters = kmeans.predict(gmm_X)
        cluster_centers = kmeans.cluster_centers_
    else:
        print(f"Selected GMM configuration: {best_config}")
        clusters = best_gmm.predict(gmm_X)
        
        # Get the means of the Gaussian components
        cluster_centers = best_gmm.means_
    
    # Find average L value (lightness) for each cluster
    l_index = feature_cols.index('avg_l')
    a_index = feature_cols.index('avg_a')
    b_index = feature_cols.index('avg_b')
    
    # Now use a different score for mapping to skin types
    skin_tone_scores = []
    for center in cluster_centers:
        # Invert L value so higher value = darker skin
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]         # a can be negative (green) or positive (red)
        b_value = center[b_index]         # b can be negative (blue) or positive (yellow)
        
        # Use a different formula from K-means to get different results
        # Give higher weight to undertones (a and b values)
        score = (0.5 * l_value) + (0.3 * a_value) + (0.2 * b_value)
        skin_tone_scores.append(score)
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    sorted_indices = np.argsort(skin_tone_scores)
    
    # Map original clusters to Monk skin types (1-10)
    # If we have more than 10 components, we need to map them to 10 types
    cluster_to_skin_type = {}
    
    if len(cluster_centers) <= 10:
        # Direct mapping for 10 or fewer clusters
        for skin_type, cluster in enumerate(sorted_indices, 1):
            cluster_to_skin_type[cluster] = skin_type
    else:
        # For more than 10 clusters, map them to 10 skin types
        # Create equally spaced bins
        n_clusters = len(cluster_centers)
        step = n_clusters / 10
        
        for i, cluster in enumerate(sorted_indices):
            skin_type = min(10, max(1, int(1 + i / step)))
            cluster_to_skin_type[cluster] = skin_type
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in clusters]
    
    # Print distribution
    distribution = results_df['predicted_skin_type'].value_counts().sort_index()
    print("Final GMM distribution:")
    print(distribution)
    
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
    #X = (X - X.mean()) / X.std()
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    l_index = feature_cols.index('avg_l')
    
    # Apply a non-linear transformation to L values to compress extremes and emphasize midrange
    # This transformation is applied to the already scaled data
    X_transformed = X.copy()
    X_transformed.iloc[:, l_index] = np.tanh(X.iloc[:, l_index] * 0.5)  # Compress extremes, preserve middle

    
    
    # Use Spectral Clustering
    print("Running Spectral Clustering...")
    spectral = SpectralClustering(
        n_clusters=10,
        random_state=42,
        affinity='kmeans',  
        gamma=0.1,  # Adjust gamma for more natural clustering
        n_neighbors=20,
        assign_labels='discretize',
        n_init=50
    )
    clusters = spectral.fit_predict(X_transformed)
    
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
        score = (0.6 * l_value) + (0.2 * a_value) + (0.2 * b_value)
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
    Classify skin types using DBSCAN clustering with improved parameters.
    
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
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Try different DBSCAN parameters until we get a reasonable number of clusters
    # Start with a higher eps value and adjust if needed
    eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_samples_values = [2, 3, 4, 5]
    
    best_n_clusters = 0
    best_clusters = None
    
    print("Tuning DBSCAN parameters...")
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Try DBSCAN with these parameters
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_norm)
            
            # Count unique clusters (excluding noise points)
            unique_clusters = np.unique(clusters)
            n_clusters = len([c for c in unique_clusters if c != -1])
            
            print(f"  eps={eps}, min_samples={min_samples}: {n_clusters} clusters")
            
            # If this is better than our current best, update it
            if n_clusters > best_n_clusters and n_clusters <= 10:
                best_n_clusters = n_clusters
                best_clusters = clusters.copy()
                
            # If we found a good number of clusters, we can stop
            if 8 <= n_clusters <= 10:
                break
        
        # If we found a good number of clusters, we can stop
        if 8 <= best_n_clusters <= 10:
            break
    
    # If we couldn't find a good number of clusters, use KMeans instead
    if best_n_clusters < 5:
        print(f"DBSCAN found only {best_n_clusters} clusters. Falling back to KMeans.")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_norm)
        cluster_centers = kmeans.cluster_centers_
    else:
        print(f"DBSCAN found {best_n_clusters} clusters.")
        clusters = best_clusters
        
        # Compute cluster centers manually
        cluster_centers = []
        valid_clusters = sorted([c for c in np.unique(clusters) if c != -1])
        
        for i in valid_clusters:
            indices = np.where(clusters == i)[0]
            center = X.iloc[indices].mean().values
            cluster_centers.append(center)
            
        # If we don't have 10 clusters, add dummy centers for the missing ones
        while len(cluster_centers) < 10:
            center = np.zeros(len(feature_cols))
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
    
    # Handle noise points first - assign a reasonable skin type (middle of range)
    if -1 in np.unique(clusters):
        cluster_to_skin_type[-1] = 5
    
    # Map existing clusters to skin types
    valid_clusters = sorted([c for c in np.unique(clusters) if c != -1])
    for skin_type, i in enumerate(sorted_indices[:len(valid_clusters)], 1):
        if i < len(valid_clusters):
            cluster_id = valid_clusters[i % len(valid_clusters)]
            cluster_to_skin_type[cluster_id] = skin_type
    
    # Handle any unmapped clusters
    for c in valid_clusters:
        if c not in cluster_to_skin_type:
            # Assign based on cluster index
            cluster_to_skin_type[c] = (c % 10) + 1
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type.get(c, ((c+1) % 10) + 1) for c in clusters]
    
    return results_df

def classify_with_agglomerative(features_df, output_folder):
    """
    Classify skin types using Agglomerative Clustering with improved balance.
    
    Args:
        features_df: DataFrame with extracted skin features
        output_folder: Folder to save results
        
    Returns:
        DataFrame with added skin type predictions
    """
    # Import the necessary classes
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Features to use for clustering
    feature_cols = [col for col in features_df.columns if col.startswith(('avg_', 'std_', 'dom'))]
    
    # Create feature matrix
    X = features_df[feature_cols].copy()
    
    # Try different scaling approaches to get more balanced results
    # 1. Standard scaling
    X_std = StandardScaler().fit_transform(X)
    # 2. Min-max scaling
    X_minmax = MinMaxScaler().fit_transform(X)
    
    # Helper function to evaluate distribution balance
    def evaluate_distribution(clusters):
        counts = np.bincount(clusters, minlength=10)
        # Calculate coefficient of variation (lower is more balanced)
        cv = np.std(counts) / np.mean(counts)
        return cv, counts
    
    # Try different linkage and scaling combinations
    best_cv = float('inf')
    best_clusters = None
    best_combo = None
    
    print("Trying different Agglomerative Clustering configurations...")
    for linkage in ['ward', 'complete', 'average']:
        for distance_threshold in [None, 1.5, 2.0]:
            for X_scaled, scale_name in [(X_std, 'std'), (X_minmax, 'minmax')]:
                # If using distance_threshold, don't specify n_clusters
                if distance_threshold:
                    agg = AgglomerativeClustering(
                        distance_threshold=distance_threshold,
                        n_clusters=None,
                        linkage=linkage
                    )
                else:
                    agg = AgglomerativeClustering(
                        n_clusters=10,
                        linkage=linkage
                    )
                
                clusters = agg.fit_predict(X_scaled)
                
                # If we used distance_threshold and got too few or too many clusters, skip
                if len(np.unique(clusters)) < 5 or len(np.unique(clusters)) > 15:
                    continue
                
                # If we don't have exactly 10 clusters, we'll need different handling later
                n_clusters = len(np.unique(clusters))
                
                cv, counts = evaluate_distribution(clusters)
                combo = f"{linkage}_{scale_name}"
                if distance_threshold:
                    combo += f"_dist{distance_threshold}"
                
                print(f"  {combo}: {n_clusters} clusters, CV={cv:.3f}, counts={counts}")
                
                # If this is more balanced and has reasonable number of clusters, keep it
                if cv < best_cv and 8 <= n_clusters <= 12:
                    best_cv = cv
                    best_clusters = clusters.copy()
                    best_combo = combo
    
    #After selecting the best clustering
    if best_clusters is None:
        print("Could not find a balanced clustering. Falling back to KMeans.")
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        best_clusters = kmeans.fit_predict(X_std)
        best_scaled_X = X_std  # Save the scaled data used for clustering
        best_combo = "kmeans_std"
    else:
        print(f"Selected {best_combo} with CV={best_cv:.3f}")
        # Save which scaled version was used
        best_scaled_X = X_std if "std" in best_combo else X_minmax
    
    # Get unique clusters
    unique_clusters = np.unique(best_clusters)
    n_clusters = len(unique_clusters)
    
    # If we don't have exactly 10 clusters, we might need special handling
    if n_clusters != 10:
        print(f"Found {n_clusters} clusters instead of 10. Adjusting mapping...")
        # We'll create a mapping to 10 skin types based on L values
        
    # For Agglomerative Clustering, we need to compute cluster centers manually
    cluster_centers = []
    for i in unique_clusters:
        indices = np.where(best_clusters == i)[0]
        
        if len(indices) == 0:
            center = np.zeros(len(feature_cols))
        else:
            # Use best_scaled_X to find the center in the same scaled space
            scaled_center = np.mean(best_scaled_X[indices], axis=0)
            
            # If you want centers in the original feature space, inverse transform:
            if "std" in best_combo:
                scaler = StandardScaler().fit(X)
                # Reshape for inverse_transform
                center = scaler.inverse_transform([scaled_center])[0]
            elif "minmax" in best_combo:
                scaler = MinMaxScaler().fit(X)
                center = scaler.inverse_transform([scaled_center])[0]
            else:
                center = scaled_center
        
        cluster_centers.append(center)
    
    cluster_centers = np.array(cluster_centers)
    
    # Find L, A, B indices
    l_index = feature_cols.index('avg_l')
    a_index = feature_cols.index('avg_a')
    b_index = feature_cols.index('avg_b')
    
    # Pre-sort clusters by L value (lighter to darker) before scoring
    # This gives more predictable results
    l_values = [center[l_index] for center in cluster_centers]
    pre_sort = np.argsort(l_values)[::-1]  # Descending order (higher L = lighter)
    
    # Now calculate scores that incorporate A and B
    skin_tone_scores = []
    for i, idx in enumerate(pre_sort):
        center = cluster_centers[idx]
        l_value = 100 - center[l_index]  # L is 0-100, invert for ordering
        a_value = center[a_index]
        b_value = center[b_index]
        
        # Weight L more heavily to keep the lightness-based ordering
        score = (0.4 * l_value) + (0.30 * a_value) + (0.30 * b_value)
        # Add a small bias based on pre-sorting to maintain stable ordering
        score += i * 0.01
        skin_tone_scores.append((idx, score))
    
    # Sort by score
    skin_tone_scores.sort(key=lambda x: x[1])
    
    # Create cluster_to_skin_type mapping
    cluster_to_skin_type = {}
    
    # If we have exactly 10 clusters, direct 1:1 mapping
    if n_clusters == 10:
        for skin_type, (cluster_idx, _) in enumerate(skin_tone_scores, 1):
            cluster = unique_clusters[cluster_idx]
            cluster_to_skin_type[cluster] = skin_type
    # If we have fewer than 10 clusters, some skin types will be skipped
    elif n_clusters < 10:
        # Map clusters to evenly spaced skin types
        step = 9.0 / (n_clusters - 1)  # Ensure we use full range
        for i, (cluster_idx, _) in enumerate(skin_tone_scores):
            cluster = unique_clusters[cluster_idx]
            skin_type = int(1 + i * step)
            skin_type = min(10, max(1, skin_type))  # Ensure within 1-10 range
            cluster_to_skin_type[cluster] = skin_type
    # If more than 10 clusters, some will map to the same skin type
    else:
        # Map multiple clusters to 10 skin types
        step = n_clusters / 10.0
        for i, (cluster_idx, _) in enumerate(skin_tone_scores):
            cluster = unique_clusters[cluster_idx]
            skin_type = int(1 + i / step)
            skin_type = min(10, max(1, skin_type))  # Ensure within 1-10 range
            cluster_to_skin_type[cluster] = skin_type
    
    # Apply mapping to get predicted skin types
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in best_clusters]
    
    # Print distribution of final skin types
    skin_type_counts = results_df['predicted_skin_type'].value_counts().sort_index()
    print("Final distribution of skin types:")
    print(skin_type_counts)
    
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

def process_with_multiple_methods(csv_path, image_folder, output_folder, features_df=None, max_images=None, save_debug_masks=True, methods=None):
    """
    Process the dataset using multiple clustering methods.
    
    Args:
        csv_path: Path to CSV file
        image_folder: Path to image folder
        output_folder: Base output folder
        features_df: Pre-extracted features (if None, will extract features)
        max_images: Maximum number of images to process
        save_debug_masks: Whether to save mask visualizations
        methods: Dictionary of methods to run (if None, runs all methods)
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
    
    # Define all available clustering methods
    all_methods = {
        'kmeans': classify_with_kmeans,
        'gmm': classify_with_gmm,
        'spectral': classify_with_spectral,
        'dbscan': classify_with_dbscan,
        'agglomerative': classify_with_agglomerative
    }
    
    # Use the specified methods or all methods
    clustering_methods = methods if methods is not None else all_methods
    
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
    
    # Create comparison visualization if we have more than one method
    if len(results) > 1:
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
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'red']
    
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
    max_images = 500  # Set to None for all images
    
    # Specify which methods to run (comment out any you don't want to run)
    methods_to_run = {
        'kmeans': classify_with_kmeans,
        #'gmm': classify_with_gmm,
        #'spectral': classify_with_spectral,
        #'dbscan': classify_with_dbscan,  
        'agglomerative': classify_with_agglomerative 
    }
    
    # Process with selected clustering methods
    results = process_with_multiple_methods(
        csv_path, 
        image_folder, 
        output_folder, 
        features_df=None, 
        max_images=max_images,
        save_debug_masks=True,
        methods=methods_to_run  # Pass the selected methods
    )
    
    if results:
        print("\nClustering comparison complete!")
    else:
        print("Clustering comparison failed.")

if __name__ == "__main__":
    main()