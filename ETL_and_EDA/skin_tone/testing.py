import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch

# Set global variables
MONK_SKIN_TYPES = 10  # 10 levels from lightest (1) to darkest (10)

def crop_black_borders(img):
    """
    Crop black borders from an image.
    
    Args:
        img: Input image
        
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
        if np.mean(gray[i, :]) > 10:  # Threshold for near-black
            top_line = i
            break
    
    # Find bottom crop line
    bottom_line = height - 1
    for i in range(height - 1, -1, -1):
        if np.mean(gray[i, :]) > 10:
            bottom_line = i
            break
    
    # Find left crop line
    left_line = 0
    for i in range(width):
        if np.mean(gray[:, i]) > 10:
            left_line = i
            break
    
    # Find right crop line
    right_line = width - 1
    for i in range(width - 1, -1, -1):
        if np.mean(gray[:, i]) > 10:
            right_line = i
            break
    
    # If entire image is black or cropping would remove everything, return original
    if top_line >= bottom_line or left_line >= right_line:
        return img
    
    # Crop the image
    cropped_img = img[top_line:bottom_line + 1, left_line:right_line + 1]
    
    return cropped_img

def create_skin_mask(img):
    """
    Create a mask that identifies normal skin regions and excludes lesions.
    
    Args:
        img: Input image
        
    Returns:
        mask: Boolean mask where True indicates probable skin regions
    """
    # Convert to LAB and HSV color spaces
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    
    # Create masks based on color characteristics
    # Skin is typically lighter, less red, and less saturated than lesions
    
    # Lightness mask - keep pixels that are lighter
    l_mean = np.mean(l_channel)
    l_std = np.std(l_channel)
    l_mask = l_channel > (l_mean - 0.25 * l_std)
    
    # Redness mask - keep pixels that are less red
    a_mean = np.mean(a_channel)
    a_std = np.std(a_channel)
    a_mask = a_channel < (a_mean + 0.25 * a_std)
    
    # Saturation mask - keep pixels that are less saturated
    s_mean = np.mean(s_channel)
    s_std = np.std(s_channel)
    s_mask = s_channel < (s_mean + 0.25 * s_std)
    
    # Edge detection to identify lesion boundaries
    edges = cv2.Canny(l_channel.astype(np.uint8), 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    edge_mask = cv2.dilate(edges, kernel, iterations=1)
    edge_mask = edge_mask > 0
    
    # Create border mask - assume image edges are more likely to be normal skin
    h, w = img.shape[:2]
    border_width = int(min(h, w) * 0.15)  # 15% border width
    
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_width, :] = True  # Top border
    border_mask[-border_width:, :] = True  # Bottom border
    border_mask[:, :border_width] = True  # Left border
    border_mask[:, -border_width:] = True  # Right border
    
    # Combine masks
    # 1. Color-based mask
    color_mask = l_mask & a_mask & s_mask
    
    # 2. Final mask
    final_mask = (color_mask | border_mask) & ~edge_mask
    
    # Check if the mask is too restrictive
    if np.sum(final_mask) < (img.size * 0.05):  # Less than 5% of the image
        # Fall back to just the lightness mask which is usually most reliable
        final_mask = l_mask
        
        # If still too restrictive, use entire image
        if np.sum(final_mask) < (img.size * 0.05):
            final_mask = np.ones_like(l_mask)
    
    return final_mask

def extract_skin_features(image_path):
    """
    Extract features related to skin tone.
    
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
        
        # Crop black borders
        img = crop_black_borders(img)
        
        # Create mask to identify skin regions
        skin_mask = create_skin_mask(img)
        
        # Convert to LAB color space
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        # Convert to HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_img)
        
        # Extract values from masked regions
        l_values = l_channel[skin_mask]
        a_values = a_channel[skin_mask]
        b_values = b_channel[skin_mask]
        h_values = h_channel[skin_mask]
        s_values = s_channel[skin_mask]
        v_values = v_channel[skin_mask]
        
        # If we have too few pixels, fall back to entire image
        if len(l_values) < 100:
            l_values = l_channel.flatten()
            a_values = a_channel.flatten()
            b_values = b_channel.flatten()
            h_values = h_channel.flatten()
            s_values = s_channel.flatten()
            v_values = v_channel.flatten()
        
        # Focus on the brightest pixels (likely to be normal skin)
        if len(l_values) > 200:
            # Take top 60% brightest pixels
            brightness_threshold = np.percentile(l_values, 40)
            brightest_indices = l_values >= brightness_threshold
            
            l_values = l_values[brightest_indices]
            a_values = a_values[brightest_indices]
            b_values = b_values[brightest_indices]
            h_values = h_values[brightest_indices]
            s_values = s_values[brightest_indices]
            v_values = v_values[brightest_indices]
        
        # Calculate features
        features = {}
        
        # Basic color statistics
        features['avg_l'] = np.mean(l_values)
        features['std_l'] = np.std(l_values)
        features['med_l'] = np.median(l_values)
        features['q25_l'] = np.percentile(l_values, 25)
        features['q75_l'] = np.percentile(l_values, 75)
        
        features['avg_a'] = np.mean(a_values)
        features['std_a'] = np.std(a_values)
        features['med_a'] = np.median(a_values)
        
        features['avg_b'] = np.mean(b_values)
        features['std_b'] = np.std(b_values)
        features['med_b'] = np.median(b_values)
        
        # HSV features
        features['avg_h'] = np.mean(h_values)
        features['avg_s'] = np.mean(s_values)
        features['avg_v'] = np.mean(v_values)
        
        # Derived features
        features['skin_tone_index'] = (100 - features['avg_l']) + (features['avg_a'] * 0.3) + (features['avg_b'] * 0.3)
        features['undertone_ratio'] = features['avg_a'] / (features['avg_b'] + 1e-5)  # Avoid division by zero
        
        # Identify dominant colors using K-means
        lab_pixels = np.column_stack((l_values, a_values, b_values))
        
        # Handle case where there are fewer pixels than clusters
        n_clusters = min(3, len(lab_pixels))
        if n_clusters < 2:
            return features
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(lab_pixels)
        
        # Get dominant colors and their frequencies
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        sorted_colors = colors[sorted_indices]
        
        # Add dominant color features
        for i in range(min(3, len(sorted_colors))):
            features[f'dom{i+1}_l'] = sorted_colors[i][0]
            features[f'dom{i+1}_a'] = sorted_colors[i][1]
            features[f'dom{i+1}_b'] = sorted_colors[i][2]
            features[f'dom{i+1}_freq'] = counts[sorted_indices[i]] / len(labels)
        
        return features
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_features_parallel(image_paths, n_jobs=None):
    """
    Extract features from multiple images in parallel.
    
    Args:
        image_paths: List of image paths
        n_jobs: Number of processes to use (None = use CPU count)
        
    Returns:
        List of feature dictionaries
    """
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Extracting features using {n_jobs} processes...")
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(extract_skin_features, image_paths), total=len(image_paths)))
    
    # Filter out None results
    return [r for r in results if r is not None]

def cluster_with_kmeans(features_df, n_clusters=MONK_SKIN_TYPES):
    """
    Cluster skin features using K-means.
    
    Args:
        features_df: DataFrame with skin features
        n_clusters: Number of clusters (default is MONK_SKIN_TYPES)
        
    Returns:
        DataFrame with added cluster labels
    """
    print("Clustering with K-means...")
    
    # Select features for clustering
    feature_cols = [col for col in features_df.columns if col not in ['image_name']]
    X = features_df[feature_cols].copy()
    
    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters if not specified
    if n_clusters is None:
        best_score = -float('inf')
        best_n = 2
        
        for n in range(2, min(11, len(X_scaled) // 5)):  # Try up to 10 clusters
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            try:
                score = silhouette_score(X_scaled, clusters)
                print(f"  K-means with {n} clusters: silhouette = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_n = n
            except:
                continue
        
        n_clusters = best_n
        print(f"Selected {n_clusters} clusters based on silhouette score")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate clustering quality metrics
    try:
        silhouette = silhouette_score(X_scaled, clusters)
        calinski = calinski_harabasz_score(X_scaled, clusters)
        print(f"K-means clustering metrics - Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}")
    except:
        print("Could not calculate clustering metrics")
    
    # Map clusters to Monk skin types
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    # Calculate a skin tone score for each center
    # We need to transform the centers back to original feature space
    centers_original = scaler.inverse_transform(centers)
    
    # Find indices of relevant features
    l_indices = [i for i, col in enumerate(feature_cols) if 'avg_l' in col or 'med_l' in col or 'dom' in col and 'l' in col]
    a_indices = [i for i, col in enumerate(feature_cols) if 'avg_a' in col or 'med_a' in col or 'dom' in col and 'a' in col]
    b_indices = [i for i, col in enumerate(feature_cols) if 'avg_b' in col or 'med_b' in col or 'dom' in col and 'b' in col]
    
    skin_tone_scores = []
    for i, center in enumerate(centers_original):
        # Calculate weighted scores
        l_score = np.mean([100 - center[j] for j in l_indices])  # Invert L value (higher = darker)
        a_score = np.mean([center[j] for j in a_indices])        # Higher a = more red
        b_score = np.mean([center[j] for j in b_indices])        # Higher b = more yellow
        
        # Combined skin tone score
        score = (0.7 * l_score) + (0.15 * a_score) + (0.15 * b_score)
        skin_tone_scores.append((i, score))
    
    # Sort clusters by skin tone score (lower to higher = lighter to darker)
    skin_tone_scores.sort(key=lambda x: x[1])
    
    # Map clusters to skin types (1-10)
    cluster_to_skin_type = {}
    
    if n_clusters <= MONK_SKIN_TYPES:
        # Spread clusters across range
        step = (MONK_SKIN_TYPES - 1) / (n_clusters - 1) if n_clusters > 1 else 1
        for rank, (cluster, _) in enumerate(skin_tone_scores):
            skin_type = int(1 + rank * step)
            cluster_to_skin_type[cluster] = min(MONK_SKIN_TYPES, max(1, skin_type))
    else:
        # Map multiple clusters to same skin type if needed
        step = n_clusters / MONK_SKIN_TYPES
        for rank, (cluster, _) in enumerate(skin_tone_scores):
            skin_type = int(1 + rank / step)
            cluster_to_skin_type[cluster] = min(MONK_SKIN_TYPES, max(1, skin_type))
    
    # Add skin type column
    results_df = features_df.copy()
    results_df['predicted_skin_type'] = [cluster_to_skin_type[c] for c in clusters]
    results_df['cluster_label'] = clusters
    
    return results_df, kmeans, scaler

def train_prediction_model(features_df, output_folder):
    """
    Train a model to predict Monk skin type from features.
    
    Args:
        features_df: DataFrame with features and skin type labels
        output_folder: Folder to save the model
        
    Returns:
        Trained model
    """
    
    
    print("Training prediction model...")
    
    # Select features and target
    feature_cols = [col for col in features_df.columns if col not in 
                   ['image_name', 'predicted_skin_type', 'cluster_label']]
    X = features_df[feature_cols].copy()
    y = features_df['predicted_skin_type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    print("Model evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_importance.png'), dpi=300)
    plt.close()
    
    # Save the model in multiple formats
    
    # 1. Save as joblib
    #model_path = os.path.join(project_root, 'trained_model', 'skin_type_predictor.joblib')
    #joblib.dump(model, model_path)
    
    # 2. Save as pytorch model (.pth)
    # Create project-wide model directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(project_root, 'trained_model')
    os.makedirs(model_dir, exist_ok=True)
    
    # Convert to PyTorch format (serializing key parameters)
    model_dict = {
        'n_estimators': model.n_estimators,
        'max_features': model.max_features,
        'feature_names': feature_cols,
        'n_classes': len(np.unique(y)),
        'feature_importances': model.feature_importances_,
        'model_type': 'RandomForest',
        'classes': model.classes_.tolist()
    }
    
    # Save as .pth file
    torch_path = os.path.join(model_dir, 'skin_type_predictor.pth')
    torch.save(model_dict, torch_path)
    
    print(f"Model saved to:")
    #print(f"  - {model_path} (joblib format)")
    print(f"  - {torch_path} (PyTorch format)")
    
    return model

def predict_skin_type(image_path, model, scaler):
    """
    Predict Monk skin type for a single image.
    
    Args:
        image_path: Path to image
        model: Trained model
        scaler: Feature scaler
        
    Returns:
        Predicted skin type (1-10)
    """
    # Extract features
    features = extract_skin_features(image_path)
    
    if features is None:
        print(f"Could not extract features from {image_path}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Select features used by the model
    feature_cols = model.feature_names_in_
    
    # Ensure all required columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Scale features
    X = scaler.transform(df[feature_cols])
    
    # Predict
    skin_type = model.predict(X)[0]
    
    return skin_type

def visualize_skin_type_distribution(results_df, output_folder):
    """
    Visualize the distribution of skin types.
    
    Args:
        results_df: DataFrame with skin type predictions
        output_folder: Folder to save visualizations
    """
    print("Creating visualizations...")
    
    # Create a counter for skin types
    skin_type_counts = results_df['predicted_skin_type'].value_counts().sort_index()
    
    # Plot the distribution as a bar chart
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=skin_type_counts.index, y=skin_type_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(skin_type_counts.values):
        percentage = 100 * count / len(results_df)
        ax.text(i, count + 5, f"{count}\n({percentage:.1f}%)", ha='center')
    
    plt.title('Distribution of Monk Skin Types (Bar Chart)', fontsize=15)
    plt.xlabel('Monk Skin Type (1 = Lightest, 10 = Darkest)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, 'skin_type_distribution_bar.png'), dpi=300)
    plt.close()
    
    # Plot the distribution as a line plot
    plt.figure(figsize=(12, 6))
    plt.plot(skin_type_counts.index, skin_type_counts.values, 'o-', linewidth=2, markersize=10)
    
    # Add count labels above points
    for i, count in enumerate(skin_type_counts.values):
        percentage = 100 * count / len(results_df)
        plt.text(skin_type_counts.index[i], count + 5, f"{count}\n({percentage:.1f}%)", ha='center')
    
    plt.title('Distribution of Monk Skin Types (Line Plot)', fontsize=15)
    plt.xlabel('Monk Skin Type (1 = Lightest, 10 = Darkest)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(skin_type_counts.index)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, 'skin_type_distribution_line.png'), dpi=300)
    plt.close()
    
    # Plot the distribution as a pie chart
    plt.figure(figsize=(10, 10))
    patches, texts, autotexts = plt.pie(
        skin_type_counts.values, 
        labels=[f"Type {i}" for i in skin_type_counts.index],
        autopct='%1.1f%%',
        startangle=90,
        shadow=False,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        textprops={'fontsize': 12}
    )
    for text in autotexts:
        text.set_fontsize(10)
    
    plt.axis('equal')
    plt.title('Distribution of Monk Skin Types (Pie Chart)', fontsize=15)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, 'skin_type_distribution_pie.png'), dpi=300)
    plt.close()
    
    # Save distribution to CSV
    dist_df = pd.DataFrame({
        'skin_type': skin_type_counts.index,
        'count': skin_type_counts.values,
        'percentage': 100 * skin_type_counts.values / len(results_df)
    })
    dist_df.to_csv(os.path.join(output_folder, 'skin_type_distribution.csv'), index=False)
    
    # Create a detailed distribution table and save as image
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    table_data = [
        ['Skin Type', 'Count', 'Percentage'],
        *[[f'Type {row.skin_type}', f'{row.count}', f'{row.percentage:.1f}%'] for _, row in dist_df.iterrows()]
    ]
    table = plt.table(
        cellText=table_data,
        colWidths=[0.3, 0.3, 0.3],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    plt.title('Monk Skin Type Distribution Table', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'skin_type_distribution_table.png'), dpi=300)
    plt.close()
    
    # Additional visualization: L value boxplots by skin type
    plt.figure(figsize=(14, 6))
    sns.boxplot(x='predicted_skin_type', y='avg_l', data=results_df)
    plt.title('L Value Distribution by Skin Type', fontsize=15)
    plt.xlabel('Monk Skin Type', fontsize=12)
    plt.ylabel('Average L Value (Lightness)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, 'l_value_by_skin_type.png'), dpi=300)
    plt.close()
    
    # Feature correlation with skin type
    feature_cols = [col for col in results_df.columns if col not in 
                  ['image_name', 'predicted_skin_type', 'cluster_label']]
    
    correlations = []
    for feature in feature_cols:
        corr = np.corrcoef(results_df['predicted_skin_type'], results_df[feature])[0, 1]
        correlations.append((feature, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Plot top 15 correlations
    top_features = [x[0] for x in correlations[:15]]
    top_corrs = [x[1] for x in correlations[:15]]
    
    plt.figure(figsize=(12, 8))
    colors = ['#2b83ba' if c > 0 else '#d7191c' for c in top_corrs]
    plt.barh(top_features, top_corrs, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Feature Correlation with Monk Skin Type', fontsize=15)
    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, 'feature_correlation.png'), dpi=300)
    plt.close()

def create_sample_grid(results_df, image_folder, output_folder, grid_shape=(5, 5)):
    """
    Create a grid of sample images for each skin type.
    
    Args:
        results_df: DataFrame with skin type predictions
        image_folder: Folder containing the images
        output_folder: Folder to save visualizations
        grid_shape: Shape of the sample grid (rows, cols)
    """
    print("Creating sample image grids...")
    
    # Create folder for samples
    samples_folder = os.path.join(output_folder, 'sample_images')
    os.makedirs(samples_folder, exist_ok=True)
    
    # Calculate grid size
    rows, cols = grid_shape
    n_samples = rows * cols
    
    # Create a grid for each skin type
    for skin_type in range(1, MONK_SKIN_TYPES + 1):
        # Get images for this skin type
        type_imgs = results_df[results_df['predicted_skin_type'] == skin_type]
        
        if len(type_imgs) == 0:
            print(f"No images for skin type {skin_type}")
            continue
        
        # Sample images
        if len(type_imgs) <= n_samples:
            samples = type_imgs
        else:
            samples = type_imgs.sample(n_samples, random_state=42)
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle(f'Sample Images - Monk Skin Type {skin_type}', fontsize=16)
        
        # If we have fewer samples than the grid size
        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')
        
        # Display samples
        for i, (_, sample) in enumerate(samples.iterrows()):
            if i >= len(axes):
                break
                
            image_name = sample['image_name']
            
            # Find the image file
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = os.path.join(image_folder, f"{image_name}{ext}")
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            
            if image_path is None:
                continue
                
            # Load and display image
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].set_title(f"ID: {image_name}", fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(samples_folder, f'skin_type_{skin_type}_samples.png'), dpi=300)
        plt.close()

def save_debug_image(image_path, output_folder, max_images=100):
    """
    Save a debug image showing the skin mask.
    
    Args:
        image_path: Path to the image
        output_folder: Folder to save debug images
        max_images: Maximum number of debug images to create
    """
    # Counter for limiting debug images
    if not hasattr(save_debug_image, 'count'):
        save_debug_image.count = 0
    
    if save_debug_image.count >= max_images:
        return
        
    try:
        # Create debug folder
        debug_folder = os.path.join(output_folder, 'debug')
        os.makedirs(debug_folder, exist_ok=True)
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return
            
        # Crop black borders
        img = crop_black_borders(img)
        
        # Create skin mask
        skin_mask = create_skin_mask(img)
        
        # Convert to LAB
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        
        # Create visualizations
        l_viz = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        a_viz = cv2.normalize(a_channel, None, 0, 255, cv2.NORM_MINMAX)
        b_viz = cv2.normalize(b_channel, None, 0, 255, cv2.NORM_MINMAX)
        
        l_viz_color = cv2.cvtColor(l_viz, cv2.COLOR_GRAY2BGR)
        a_viz_color = cv2.cvtColor(a_viz, cv2.COLOR_GRAY2BGR)
        b_viz_color = cv2.cvtColor(b_viz, cv2.COLOR_GRAY2BGR)
        
        # Create skin mask visualization
        skin_viz = img.copy()
        skin_viz[~skin_mask] = [0, 0, 0]
        
        # Create non-skin mask visualization
        non_skin_viz = img.copy()
        non_skin_viz[skin_mask] = [0, 0, 0]
        
        # Create edge visualization
        edges = cv2.Canny(l_channel.astype(np.uint8), 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edge_mask = cv2.dilate(edges, kernel, iterations=1)
        
        edge_viz = np.zeros_like(img)
        edge_viz[edge_mask > 0] = [0, 0, 255]  # Red for edges
        
        # Combine visualizations
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create a grid of visualizations
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f'Skin Detection Debug: {img_name}', fontsize=14)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # L channel
        axes[0, 1].imshow(cv2.cvtColor(l_viz_color, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("L Channel (Lightness)")
        axes[0, 1].axis('off')
        
        # A channel
        axes[0, 2].imshow(cv2.cvtColor(a_viz_color, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("A Channel (Green-Red)")
        axes[0, 2].axis('off')
        
        # B channel
        axes[1, 0].imshow(cv2.cvtColor(b_viz_color, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("B Channel (Blue-Yellow)")
        axes[1, 0].axis('off')
        
        # Detected skin
        axes[1, 1].imshow(cv2.cvtColor(skin_viz, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Detected Skin Regions")
        axes[1, 1].axis('off')
        
        # Edges
        axes[1, 2].imshow(cv2.cvtColor(edge_viz, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Detected Edges")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_folder, f'{img_name}_debug.png'), dpi=300)
        plt.close()
        
        save_debug_image.count += 1
        
    except Exception as e:
        print(f"Error creating debug image for {image_path}: {e}")

def main(csv_path, image_folder, output_folder, n_images=None, save_debug=True, clean_output=False):
    """
    Main function to run skin type clustering on ISIC 2020 images.
    
    Args:
        csv_path: Path to original CSV file
        image_folder: Folder containing images
        output_folder: Folder to save results
        n_images: Number of images to process (None = all)
        save_debug: Whether to save debug images
        clean_output: Whether to clean output directory before starting
    
    Returns:
        DataFrame with added skin type predictions
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Clean output directory if requested
    if clean_output:
        print(f"Cleaning output directory: {output_folder}")
        for item in os.listdir(output_folder):
            item_path = os.path.join(output_folder, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        print("Output directory cleaned")
    
    print(f"Processing ISIC 2020 images for Monk skin type classification...")
    
    # Load original CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} entries")
    
    # Limit to specified number of images
    if n_images is not None and n_images < len(df):
        df = df.sample(n_images, random_state=42)
        print(f"Selected {len(df)} random images")
    
    # Get image paths
    image_paths = []
    found_images = 0
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        
        # Check for different extensions
        found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            path = os.path.join(image_folder, f"{image_name}{ext}")
            if os.path.exists(path):
                image_paths.append(path)
                found = True
                found_images += 1
                break
        
        if not found:
            print(f"Warning: Could not find image for {image_name}")
    
    print(f"Found {found_images} out of {len(df)} images")
    
    # Check for existing features file
    features_path = os.path.join(output_folder, 'skin_features.csv')
    
    if os.path.exists(features_path):
        print(f"Loading existing features from {features_path}")
        features_df = pd.read_csv(features_path)
    else:
        # Extract features
        feature_results = extract_features_parallel(image_paths)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_results)
        
        # Extract image names from paths
        image_names = []
        for i, result in enumerate(feature_results):
            if i < len(image_paths):
                image_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                image_names.append(image_name)
            
        # Add image names to dataframe
        features_df['image_name'] = image_names
        
        # Save features
        features_df.to_csv(features_path, index=False)
        print(f"Saved extracted features to {features_path}")
    
    # Save debug images if requested
    if save_debug:
        # Randomly sample images for debugging (up to 100)
        sample_size = min(100, len(image_paths))
        sample_paths = np.random.choice(image_paths, sample_size, replace=False)
        print(f"Generating debug images for {sample_size} images...")
        for path in tqdm(sample_paths):
            save_debug_image(path, output_folder)
    
    # Perform clustering
    results_df, kmeans, scaler = cluster_with_kmeans(features_df)
    
    # Save clustered results
    results_path = os.path.join(output_folder, 'skin_type_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved clustering results to {results_path}")
    
    # Save models
    joblib.dump(kmeans, os.path.join(output_folder, 'kmeans_model.joblib'))
    joblib.dump(scaler, os.path.join(output_folder, 'feature_scaler.joblib'))
    
    # Train prediction model
    train_prediction_model(results_df, output_folder)
    
    # Merge with original CSV
    merged_df = pd.merge(
        df, 
        results_df[['image_name', 'predicted_skin_type']], 
        on='image_name'
    )
    
    # Save final results
    final_path = os.path.join(output_folder, 'ISIC_2020_with_monk_skin_types.csv')
    merged_df.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")
    
    # Create visualizations
    visualize_skin_type_distribution(results_df, output_folder)
    create_sample_grid(results_df, image_folder, output_folder)
    
    print("Monk skin type clustering complete!")
    
    return merged_df

if __name__ == "__main__":
    import argparse
    
    # =====================================================================
    # CONFIGURATION SECTION - Edit these values when running from VS Code
    # =====================================================================
    # Set default values here for direct execution in VS Code
    n_images = 3000           # Number of images to process
    save_debug = True        # Whether to save debug images
    clean_output = True      # Whether to clean output directory before starting
    # =====================================================================
    
    # Use project directory structure to locate files automatically
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define default paths based on project structure
    data_dir = os.path.join(project_root, 'data')
    default_csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    default_image_folder = os.path.join(data_dir, 'train_224X224')
    default_output_folder = os.path.join(data_dir, 'skin_type_analysis', 'clustering_comparison', 'testing')
    
    # The command-line argument code below is OPTIONAL and can be commented out
    # if you always run the script directly from VS Code
    # =====================================================================
    """
    # Command-line argument handling (can be commented out if not needed)
    parser = argparse.ArgumentParser(description="Monk Skin Type Clustering for ISIC 2020")
    parser.add_argument("--csv", help="Path to original ISIC 2020 CSV file (optional)")
    parser.add_argument("--images", help="Path to folder containing images (optional)")
    parser.add_argument("--output", help="Path to output folder (optional)")
    parser.add_argument("--n_images", type=int, default=3000, help="Number of images to process (default: 3000)")
    parser.add_argument("--no_debug", action="store_true", help="Disable saving debug images")
    parser.add_argument("--clean", action="store_true", help="Clean output directory before starting")
    
    args = parser.parse_args()
    
    # Use provided arguments or fall back to defaults
    csv_path = args.csv if args.csv else default_csv_path
    image_folder = args.images if args.images else default_image_folder
    output_folder = args.output if args.output else default_output_folder
    n_images = args.n_images
    save_debug = not args.no_debug
    clean_output = args.clean
    """
    # =====================================================================
    
    # Set paths for execution
    csv_path = default_csv_path
    image_folder = default_image_folder
    output_folder = default_output_folder
    
    print(f"Using paths:")
    print(f"  CSV: {csv_path}")
    print(f"  Images: {image_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Number of images: {n_images}")
    print(f"  Debug images: {save_debug}")
    print(f"  Clean output directory: {clean_output}")
    
    # Run the main function
    main(csv_path, image_folder, output_folder, n_images, save_debug, clean_output)