#!/usr/bin/env python3
"""
Visualization module for Monk skin type classification.
This module handles the creation of visualizations for skin features and results.
"""

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from feature_extraction import crop_black_borders, create_skin_mask

# Set global variables
MONK_SKIN_TYPES = 10  # 10 levels from lightest (1) to darkest (10)

def visualize_skin_type_distribution(results_df, model_plots_output, output_folder, 
                                     skin_type_column='predicted_skin_type',
                                     is_cnn_result=False):
    """
    Visualize the distribution of skin types.
    
    Args:
        results_df: DataFrame with skin type predictions
        model_plots_output: Folder to save visualization plots
        output_folder: Main output folder
        skin_type_column: Column name containing skin type (default='predicted_skin_type')
        is_cnn_result: Whether this is a CNN result (different column structure)
    """
    print("Creating visualizations...")

    # Check if the specified skin type column exists
    if skin_type_column not in results_df.columns:
        print(f"Warning: Column '{skin_type_column}' not found in DataFrame.")
        skin_type_columns = [col for col in results_df.columns if 'skin_type' in col.lower()]
        if skin_type_columns:
            skin_type_column = skin_type_columns[0]
            print(f"Using '{skin_type_column}' instead.")
        else:
            print("No skin type column found. Aborting visualization.")
            return

    plots_dir = os.path.join(model_plots_output, 'distribution_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a counter for skin types
    skin_type_counts = results_df[skin_type_column].value_counts().sort_index()
    
    # Add a suffix for CNN predictions to distinguish them
    suffix = "_cnn" if is_cnn_result else ""
    
    # Plot the distribution as a bar chart
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=skin_type_counts.index, y=skin_type_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(skin_type_counts.values):
        percentage = 100 * count / len(results_df)
        ax.text(i, count + 5, f"{count}\n({percentage:.1f}%)", ha='center')
    
    title_prefix = "CNN " if is_cnn_result else ""
    plt.title(f'Distribution of {title_prefix}Monk Skin Types (Bar Chart)', fontsize=15)
    plt.xlabel('Monk Skin Type (1 = Lightest, 10 = Darkest)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, f'skin_type_distribution_bar{suffix}.png'), dpi=300)
    plt.close()
    
    # Plot the distribution as a line plot
    plt.figure(figsize=(12, 6))
    plt.plot(skin_type_counts.index, skin_type_counts.values, 'o-', linewidth=2, markersize=10)
    
    # Add count labels above points
    for i, count in enumerate(skin_type_counts.values):
        percentage = 100 * count / len(results_df)
        plt.text(skin_type_counts.index[i], count + 5, f"{count}\n({percentage:.1f}%)", ha='center')
    
    plt.title(f'Distribution of {title_prefix}Monk Skin Types (Line Plot)', fontsize=15)
    plt.xlabel('Monk Skin Type (1 = Lightest, 10 = Darkest)', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(skin_type_counts.index)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, f'skin_type_distribution_line{suffix}.png'), dpi=300)
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
    plt.title(f'Distribution of {title_prefix}Monk Skin Types (Pie Chart)', fontsize=15)
    plt.tight_layout()
    
    plt.savefig(os.path.join(plots_dir, f'skin_type_distribution_pie{suffix}.png'), dpi=300)
    plt.close()
    
    # Save distribution to CSV
    dist_df = pd.DataFrame({
        'skin_type': skin_type_counts.index,
        'count': skin_type_counts.values,
        'percentage': 100 * skin_type_counts.values / len(results_df)
    })
    dist_df.to_csv(os.path.join(output_folder, f'skin_type_distribution{suffix}.csv'), index=False)
    
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
    plt.title(f'{title_prefix}Monk Skin Type Distribution Table', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'skin_type_distribution_table{suffix}.png'), dpi=300)
    plt.close()
    
    # The following visualizations are only relevant for feature dataframes (not CNN results)
    if not is_cnn_result and 'avg_l' in results_df.columns:
        # Additional visualization: L value boxplots by skin type
        plt.figure(figsize=(14, 6))
        sns.boxplot(x=skin_type_column, y='avg_l', data=results_df)
        plt.title('L Value Distribution by Skin Type', fontsize=15)
        plt.xlabel('Monk Skin Type', fontsize=12)
        plt.ylabel('Average L Value (Lightness)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, 'l_value_by_skin_type.png'), dpi=300)
        plt.close()
        
        # Feature correlation with skin type
        feature_cols = [col for col in results_df.columns if col not in 
                      ['image_name', skin_type_column, 'cluster_label']]
        
        correlations = []
        for feature in feature_cols:
            corr = np.corrcoef(results_df[skin_type_column], results_df[feature])[0, 1]
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
        
        plt.savefig(os.path.join(plots_dir, 'feature_correlation.png'), dpi=300)
        plt.close()

def create_sample_grid(results_df, image_folder, output_folder, 
                       skin_type_column='predicted_skin_type', 
                       suffix="", grid_shape=(5, 5)):
    """
    Create a grid of sample images for each skin type.
    
    Args:
        results_df: DataFrame with skin type predictions
        image_folder: Folder containing the images
        output_folder: Folder to save visualizations
        skin_type_column: Column name containing skin type
        suffix: Suffix to add to filenames
        grid_shape: Shape of the sample grid (rows, cols)
    """
    print("Creating sample image grids...")
    
    # Check if the specified skin type column exists
    if skin_type_column not in results_df.columns:
        print(f"Warning: Column '{skin_type_column}' not found in DataFrame.")
        skin_type_columns = [col for col in results_df.columns if 'skin_type' in col.lower()]
        if skin_type_columns:
            skin_type_column = skin_type_columns[0]
            print(f"Using '{skin_type_column}' instead.")
        else:
            print("No skin type column found. Aborting sample grid creation.")
            return
    
    # Create folder for samples
    samples_folder = os.path.join(output_folder, 'sample_images')
    os.makedirs(samples_folder, exist_ok=True)
    
    # Calculate grid size
    rows, cols = grid_shape
    n_samples = rows * cols
    
    # Create a grid for each skin type
    for skin_type in range(1, MONK_SKIN_TYPES + 1):
        # Get images for this skin type
        type_imgs = results_df[results_df[skin_type_column] == skin_type]
        
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
                
                # Add more info if available
                title = f"ID: {image_name}"
                if 'cnn_confidence' in sample and not pd.isna(sample['cnn_confidence']):
                    title += f"\nConf: {sample['cnn_confidence']:.2f}"
                
                axes[i].set_title(title, fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(samples_folder, f'skin_type_{skin_type}{suffix}_samples.png'), dpi=300)
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

if __name__ == "__main__":
    # This allows testing the visualization module independently
    if len(sys.argv) > 2:
        # Generate visualizations for results
        results_path = sys.argv[1]
        output_folder = sys.argv[2]
        
        results_df = pd.read_csv(results_path)
        
        # Determine the correct skin type column
        if 'predicted_skin_type' in results_df.columns:
            skin_type_column = 'predicted_skin_type'
            is_cnn_result = False
        elif 'cnn_predicted_skin_type' in results_df.columns:
            skin_type_column = 'cnn_predicted_skin_type'
            is_cnn_result = True
        else:
            skin_type_column = None
            is_cnn_result = False
            print("Warning: No skin type column found")
        
        if skin_type_column:
            visualize_skin_type_distribution(
                results_df, 
                output_folder, 
                output_folder,
                skin_type_column=skin_type_column,
                is_cnn_result=is_cnn_result
            )
        
        # If image folder is provided, also create sample grids
        if len(sys.argv) > 3 and skin_type_column:
            image_folder = sys.argv[3]
            suffix = "_cnn" if is_cnn_result else ""
            create_sample_grid(
                results_df, 
                image_folder, 
                output_folder,
                skin_type_column=skin_type_column,
                suffix=suffix
            )
    else:
        print("Usage: python visualization.py results_csv output_folder [image_folder]")