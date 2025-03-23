#!/usr/bin/env python3
"""
Feature extraction module for Monk skin type classification.
This module handles image processing and feature extraction from skin images.
"""

import os
import numpy as np
import cv2
import sys
from sklearn.cluster import KMeans
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

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

if __name__ == "__main__":
    # This allows testing the feature extraction module independently
    if len(sys.argv) > 1:
        # Extract features from a single image
        image_path = sys.argv[1]
        features = extract_skin_features(image_path)
        if features:
            for key, value in features.items():
                print(f"{key}: {value}")
        else:
            print("Failed to extract features")
    else:
        print("Please provide an image path as an argument")