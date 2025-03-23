#!/usr/bin/env python3
"""
Clustering module for Monk skin type classification.
This module handles the clustering of skin features to determine skin types.
"""

import numpy as np
import pandas as pd
import sys
import os
import joblib
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Set global variables
MONK_SKIN_TYPES = 10  # 10 levels from lightest (1) to darkest (10)

def cluster_with_kmeans(features_df, n_clusters=MONK_SKIN_TYPES, save_model_path=None):
    """
    Cluster skin features using K-means.
    
    Args:
        features_df: DataFrame with skin features
        n_clusters: Number of clusters (default is MONK_SKIN_TYPES)
        save_model_path: Path to save the model (optional)
        
    Returns:
        DataFrame with added cluster labels, kmeans model, and scaler
    """
    print("Clustering with K-means...")
    
    # Select features for clustering
    feature_cols = [col for col in features_df.columns if col not in ['image_name']]
    #feature_cols = select_optimal_features(features_df)
    #print(f"Selected {len(feature_cols)} optimal features for clustering")
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
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=20,  # Increase from 10 to 20
        init='k-means++',  # Use k-means++ initialization
        max_iter=500,  # Increase max iterations for better convergence
    )
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
        score = (0.8 * l_score) + (0.10 * a_score) + (0.10 * b_score)
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
    
    # Save the model if a path is provided
    if save_model_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        
        # Save as joblib
        joblib_path = f"{os.path.splitext(save_model_path)[0]}.joblib"
        joblib.dump(kmeans, joblib_path)
        print(f"KMeans model saved as joblib: {joblib_path}")
        
        # Save scaler
        scaler_path = f"{os.path.splitext(joblib_path)[0]}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved as joblib: {scaler_path}")
        
        # Save as PyTorch .pth
        if save_model_path.endswith('.pth'):
            pth_path = save_model_path
        else:
            pth_path = f"{os.path.splitext(save_model_path)[0]}.pth"
            
        save_kmeans_as_pytorch(kmeans, scaler, pth_path)
    
    return results_df

def save_kmeans_as_pytorch(kmeans, scaler, output_path):
    """
    Save KMeans model as a PyTorch .pth file.
    
    Args:
        kmeans: Trained KMeans model
        scaler: Feature scaler used with the model
        output_path: Path to save the .pth file
    """
    
    # Extract essential parameters from KMeans model
    kmeans_dict = {
        'cluster_centers': kmeans.cluster_centers_,
        'n_clusters': kmeans.n_clusters,
        'inertia': kmeans.inertia_,
        'labels': kmeans.labels_.tolist() if hasattr(kmeans, 'labels_') else None,
        'model_type': 'KMeans'
    }
    
    # Add scaler parameters if available
    if scaler is not None:
        scaler_dict = {
            'scale_': scaler.scale_ if hasattr(scaler, 'scale_') else None,
            'min_': scaler.min_ if hasattr(scaler, 'min_') else None,
            'data_min_': scaler.data_min_ if hasattr(scaler, 'data_min_') else None,
            'data_max_': scaler.data_max_ if hasattr(scaler, 'data_max_') else None,
            'data_range_': scaler.data_range_ if hasattr(scaler, 'data_range_') else None,
            'scaler_type': scaler.__class__.__name__
        }
        kmeans_dict['scaler'] = scaler_dict
    
    # Save as PyTorch file
    torch.save(kmeans_dict, output_path)
    print(f"KMeans model saved as PyTorch file: {output_path}")
    
    return output_path

def get_optimal_clusters(X_scaled, max_clusters=10):
    """
    Find the optimal number of clusters using silhouette score.
    
    Args:
        X_scaled: Scaled feature matrix
        max_clusters: Maximum number of clusters to try
        
    Returns:
        Optimal number of clusters
    """
    best_score = -float('inf')
    best_n = 2
    
    for n in range(2, min(max_clusters + 1, len(X_scaled) // 5)):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        try:
            score = silhouette_score(X_scaled, clusters)
            print(f"  K-means with {n} clusters: silhouette = {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_n = n
        except:
            continue
    
    return best_n

def select_optimal_features(features_df):
    """Select the most important features for skin type clustering."""
    # Ensure these critical features are included
    critical_features = ['avg_l', 'med_l', 'avg_a', 'avg_b', 'skin_tone_index', 'undertone_ratio']
    
    # Start with critical features
    selected_features = [f for f in critical_features if f in features_df.columns]
    
    # Add other features that might be helpful
    for col in features_df.columns:
        if col not in selected_features and col != 'image_name' and (
            ('dom' in col and ('l' in col or 'a' in col or 'b' in col)) or
            'std_l' in col or 'q75_l' in col
        ):
            selected_features.append(col)
    
    return selected_features

if __name__ == "__main__":
    # This allows testing the clustering module independently
    if len(sys.argv) > 1:
        # Load features from CSV and perform clustering
        features_path = sys.argv[1]
        features_df = pd.read_csv(features_path)
        results_df, _, _ = cluster_with_kmeans(features_df)
        
        # Print cluster distribution
        print("\nSkin Type Distribution:")
        print(results_df['predicted_skin_type'].value_counts().sort_index())
    else:
        print("Please provide a features CSV path as an argument")