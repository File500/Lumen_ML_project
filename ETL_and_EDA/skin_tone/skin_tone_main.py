#!/usr/bin/env python3
"""
Main script for Monk skin type classification using CNN, with GPU selection.
This script handles clustering and CNN training/prediction, skipping Random Forest.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Import from other modules
from feature_extraction import extract_features_parallel
from clustering import cluster_with_kmeans, get_optimal_clusters
from visualization import (
    visualize_skin_type_distribution,
    create_sample_grid,
    save_debug_image
)
from skin_tone_cnn_model import (
    train_cnn_model,
    cnn_batch_predict
)
from skin_tone_RF_model import tune_random_forest, train_tuned_random_forest

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report

matplotlib.use('Agg')

def main(csv_path, image_folder, output_folder, model_output_folder, n_images=None, save_debug=False, clean_output=False, 
         model_type='cnn', gpu_id=0, batch_size=32, num_epochs=20, learning_rate=0.001, cache_images=True,
         test_size=0.2, validation_size=0.1, tune_rf=False, rf_params=None):
    """
    Main function to run skin type clustering and model training.
    
    Args:
        csv_path: Path to original CSV file
        image_folder: Folder containing images
        output_folder: Folder to save results
        model_output_folder: Folder to save model
        n_images: Number of images to process (None = all)
        save_debug: Whether to save debug images
        clean_output: Whether to clean output directory before starting
        model_type: Type of model to train ('cnn' or 'rf')
        gpu_id: ID of the GPU to use (0, 1, or 2) for CNN
        batch_size: Batch size for CNN training
        num_epochs: Number of epochs for CNN training
        learning_rate: Learning rate for optimizer
        cache_images: Whether to cache images in memory for CNN
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        tune_rf: Whether to tune RF hyperparameters
        rf_params: Dictionary of RF parameters (if not tuning)
    
    Returns:
        DataFrame with skin type predictions
    """
    # Create output folders
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_output_folder, exist_ok=True)
    
    # Clean output directory if requested
    if clean_output:
        import shutil
        print(f"Cleaning output directory: {output_folder}")
        for item in os.listdir(output_folder):
            item_path = os.path.join(output_folder, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        # Recreate the directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(model_output_folder, exist_ok=True)
        print("Output directory cleaned")

    print("\n" + "-" * 50 + "\n")
    print(f"Processing images for Monk skin type classification with {model_type.upper()} model")
    
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
    
    # First, collect paths for selected images
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

    print("\n" + "-" * 50 + "\n")
    
    # Check for existing features file
    features_path = os.path.join(output_folder, 'skin_features.csv')
    
    if os.path.exists(features_path) and not clean_output:
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

    print("\n" + "-" * 50 + "\n")

    # Check for existing clustering results
    clustering_results_path = os.path.join(output_folder, 'skin_type_clustering_results.csv')
    
    if os.path.exists(clustering_results_path) and not clean_output:
        print(f"Loading existing clustering results from {clustering_results_path}")
        results_df = pd.read_csv(clustering_results_path)
    else:
        # Either clustering not found or clean_output is True, perform clustering
        if clean_output and os.path.exists(clustering_results_path):
            print("Clean output enabled, redoing clustering...")
        else:
            print("Clustering results not found, performing clustering...")
        
        # Use fixed 10 clusters to match Monk skin types
        print("Using 10 clusters to match Monk skin type classifications...")
        results_df = cluster_with_kmeans(
            features_df,
            n_clusters=10,
            save_model_path=os.path.join(model_output_folder, 'kmeans_model.pth')
        )
        
        # Save clustering results
        results_df.to_csv(clustering_results_path, index=False)
        print(f"Saved clustering results to {clustering_results_path}")

    # Create distribution plots folder
    distribution_clustering_dir = os.path.join(model_output_folder, 'clustering_plots')
    os.makedirs(distribution_clustering_dir, exist_ok=True)
    
    print("Creating visualizations for clustering results...")
    visualize_skin_type_distribution(
        results_df, 
        distribution_clustering_dir, 
        distribution_clustering_dir,
        skin_type_column='predicted_skin_type',
        is_cnn_result=False
    )
    
    # Create sample grid of images for each skin type from clustering
    create_sample_grid(
        results_df, 
        image_folder, 
        distribution_clustering_dir,
        skin_type_column='predicted_skin_type',
        suffix="_clustering"
    )

    print("\n" + "-" * 50 + "\n")
    
    # Save debug images if requested
    if save_debug:
        print("Generating debug images...")
        # Randomly sample images for debugging (up to 100)
        sample_size = min(100, len(image_paths))
        sample_paths = np.random.choice(image_paths, sample_size, replace=False)
        
        # Prepare a progress bar for debug image generation
        for path in tqdm(sample_paths, desc="Creating debug images"):
            save_debug_image(path, output_folder, max_images=100)

    print("\n" + "-" * 50 + "\n")
    
    # Split data into train, validation, and test sets
    print("Splitting data into train, validation, and test sets...")
    
    # First split out the test set
    train_val_df, test_df = train_test_split(
        results_df, 
        test_size=test_size,
        random_state=42,
        stratify=results_df['predicted_skin_type']
    )
    
    # Then split the remaining data into train and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_size/(1-test_size),  # Adjust to get the right proportion
        random_state=42,
        stratify=train_val_df['predicted_skin_type']
    )
    
    print(f"Data split: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    
    # Save the splits for reference
    train_df.to_csv(os.path.join(output_folder, 'train_set.csv'), index=False)
    val_df.to_csv(os.path.join(output_folder, 'validation_set.csv'), index=False)
    test_df.to_csv(os.path.join(output_folder, 'test_set.csv'), index=False)

    print("\n" + "-" * 50 + "\n")
    
    # Train model based on selected type
    if model_type.lower() == 'cnn':
        # Train CNN model
        print("Training CNN model...")
        model = train_cnn_model(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            image_folder=image_folder,
            output_folder=model_output_folder,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            pretrained=True,
            gpu_id=gpu_id,
            cache_images=cache_images
        )
    elif model_type.lower() == 'rf':
        # Train Random Forest model
        print("Training Random Forest model...")
        
        if tune_rf:
            print("Tuning Random Forest hyperparameters...")
            model = tune_random_forest(train_df, val_df, model_output_folder)
        else:
            # Use provided parameters or defaults
            if rf_params is None:
                rf_params = {
                    'n_estimators': 200,
                    'max_depth': 30,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'criterion': 'gini'
                }
            
            model = train_tuned_random_forest(
                train_df, 
                model_output_folder, 
                params=rf_params, 
                val_df=val_df, 
                test_df=test_df
            )
    else:
        print(f"Error: Unknown model type '{model_type}'. Must be 'cnn' or 'rf'.")
        return None
    
    # Merge clustering results with original data
    print("Creating final dataset with clustering results...")
    merged_df = pd.merge(
        df, 
        results_df[['image_name', 'predicted_skin_type']],
        on='image_name',
        how='left'
    )
    
    # Save merged results
    merged_path = os.path.join(output_folder, 'clustered_skin_types.csv')
    merged_df.to_csv(merged_path, index=False)
    print(f"Saved clustering results to {merged_path}")

    print("\n" + "-" * 50 + "\n")
    
    # Get the full dataset if we're using a subset
    if n_images is not None:
        full_df = pd.read_csv(csv_path)
    else:
        full_df = df.copy()
    
    # Use trained model to predict on the whole dataset
    print(f"Using {model_type.upper()} model to predict on entire dataset...")
    
    if model_type.lower() == 'cnn':
        # Use CNN for prediction
        cnn_results_df = cnn_batch_predict(
            df=full_df,
            image_folder=image_folder,
            output_folder=output_folder,
            model_output_folder=model_output_folder,
            model_path=os.path.join(model_output_folder, 'cnn_model_best.pth'),
            gpu_id=gpu_id,
            batch_size=batch_size
        )
        
        # Merge CNN predictions with original data
        print("Merging CNN predictions with original data...")
        final_df = pd.merge(
            full_df,
            cnn_results_df[['image_name', 'cnn_predicted_skin_type', 'cnn_confidence']],
            on='image_name',
            how='left'
        )
        
        # Save final results
        final_path = os.path.join(output_folder, f'{model_type}_predicted_skin_types.csv')
        final_df.to_csv(final_path, index=False)
        print(f"Saved CNN prediction results to {final_path}")
        
    elif model_type.lower() == 'rf':
        # Check if RF predictions already exist
        rf_predictions_path = os.path.join(output_folder, f'{model_type}_predicted_skin_types.csv')
        
        if os.path.exists(rf_predictions_path) and not clean_output:
            print(f"Loading existing RF predictions from {rf_predictions_path}")
            final_df = pd.read_csv(rf_predictions_path)
            print(f"Loaded {len(final_df)} predictions")
        else:
            print("Predicting with Random Forest model...")
            # Select features for RF prediction
            feature_cols = [col for col in results_df.columns if col not in 
                        ['image_name', 'predicted_skin_type', 'cluster_label']]
            
            # Check if all features have already been extracted
            all_features_path = os.path.join(output_folder, 'all_skin_features.csv')
            
            if os.path.exists(all_features_path) and not clean_output:
                print(f"Loading existing features from {all_features_path}")
                all_features_df = pd.read_csv(all_features_path)
            else:
                # We need to extract features for all images
                if len(full_df) > len(features_df):
                    print(f"Need to extract features for {len(full_df) - len(features_df)} additional images")
                    
                    # Get paths for images not already processed
                    additional_paths = []
                    additional_names = []
                    
                    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Finding additional images"):
                        image_name = row['image_name']
                        
                        # Skip if already extracted
                        if image_name in features_df['image_name'].values:
                            continue
                            
                        # Check for different extensions
                        for ext in ['.jpg', '.png', '.jpeg']:
                            path = os.path.join(image_folder, f"{image_name}{ext}")
                            if os.path.exists(path):
                                additional_paths.append(path)
                                additional_names.append(image_name)
                                break
                    
                    if additional_paths:
                        print(f"Extracting features for {len(additional_paths)} additional images...")
                        additional_results = extract_features_parallel(additional_paths)
                        
                        # Convert to DataFrame
                        additional_df = pd.DataFrame(additional_results)
                        additional_df['image_name'] = additional_names
                        
                        # Combine with existing features
                        all_features_df = pd.concat([features_df, additional_df], ignore_index=True)
                        
                        # Save all features
                        all_features_path = os.path.join(output_folder, 'all_skin_features.csv')
                        all_features_df.to_csv(all_features_path, index=False)
                        print(f"Saved all features to {all_features_path}")
                    else:
                        all_features_df = features_df.copy()
                else:
                    all_features_df = features_df.copy()
            
            # Merge with full_df to get all image names
            prediction_df = pd.merge(
                full_df[['image_name']],
                all_features_df,
                on='image_name',
                how='left'
            )
            
            # Select only the feature columns needed for prediction
            X_predict = prediction_df[feature_cols].fillna(0)  # Fill NA with 0 for missing features
            
            # Make predictions
            rf_predictions = model.predict(X_predict)
            
            # Get confidence scores if possible
            try:
                rf_confidence = np.max(model.predict_proba(X_predict), axis=1)
            except:
                rf_confidence = [1.0] * len(rf_predictions)  # Default confidence
            
            # Add predictions to DataFrame
            full_df['rf_predicted_skin_type'] = rf_predictions
            full_df['rf_confidence'] = rf_confidence
            
            # Save final results
            final_path = os.path.join(output_folder, f'{model_type}_predicted_skin_types.csv')
            full_df.to_csv(final_path, index=False)
            print(f"Saved Random Forest prediction results to {final_path}")
            
            # For consistent return value
            final_df = full_df
    
    print("\n" + "-" * 50 + "\n")
    
    # Create visualizations for model predictions
    print(f"Creating visualizations for {model_type.upper()} model results...")
    visualization_output = os.path.join(model_output_folder, 'plots')
    distribution_dir = os.path.join(visualization_output, 'distribution_plots')
    os.makedirs(visualization_output, exist_ok=True)
    os.makedirs(distribution_dir, exist_ok=True)
    
    # Use the appropriate column name based on model type
    prediction_column = 'cnn_predicted_skin_type' if model_type.lower() == 'cnn' else 'rf_predicted_skin_type'
    
    # Visualize distribution of predicted skin types
    visualize_skin_type_distribution(
        final_df, 
        visualization_output, 
        output_folder,
        skin_type_column=prediction_column,
        is_cnn_result=True  # Use the same format for both models
    )
    
    # Create sample grid of images for each predicted skin type
    create_sample_grid(
        final_df, 
        image_folder, 
        output_folder,
        skin_type_column=prediction_column,
        suffix=f"_{model_type}"
    )
    
    # Compare predictions with clustering results for overlap
    print("Comparing model predictions with clustering labels...")

    available_columns = final_df.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    clustering_column = 'predicted_skin_type'
    # Ensure we have both prediction columns
    clustering_column = 'predicted_skin_type'
    if clustering_column not in available_columns:
        # Try to find it by merging with results_df
        if 'predicted_skin_type' in results_df.columns:
            print("Adding clustering results to final dataframe...")
            final_df = pd.merge(
                final_df,
                results_df[['image_name', 'predicted_skin_type']],
                on='image_name',
                how='left'
            )
        else:
            print(f"Warning: Could not find clustering column '{clustering_column}'")
            # Create a fallback comparison
            print("Skipping clustering vs model comparison due to missing columns.")
            return final_df

    # Now check if we have both columns for comparison
    if clustering_column in final_df.columns and prediction_column in final_df.columns:
        labeled_df = final_df.dropna(subset=[clustering_column, prediction_column])
    
    if len(labeled_df) > 0:
        # Get ground truth (clustering) and predictions
        y_true = labeled_df['predicted_skin_type'].astype(int).tolist()
        y_pred = labeled_df[prediction_column].astype(int).tolist()
        
        # Generate confusion matrix
        cm_all = confusion_matrix(y_true, y_pred)
        
        # Plot and save
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({model_type.upper()} vs Clustering)')
        plt.xlabel(f'Predicted by {model_type.upper()}')
        plt.ylabel('Assigned by Clustering')
        plt.savefig(os.path.join(output_folder, f'confusion_matrix_{model_type}_vs_clustering.png'))
        plt.close()
        
        # Generate and print classification report
        report_all = classification_report(y_true, y_pred)
        print(f"\nClassification Report ({model_type.upper()} vs Clustering on all labeled images):")
        print(report_all)
        
        # Save report
        with open(os.path.join(output_folder, f'classification_report_{model_type}_vs_clustering.txt'), 'w') as f:
            f.write(f"CLASSIFICATION REPORT ({model_type.upper()} vs CLUSTERING)\n")
            f.write("=======================================\n\n")
            f.write(report_all)
    
    print(f"Monk skin type classification with {model_type.upper()} complete!")
    
    return final_df

if __name__ == "__main__":
    # =====================================================================
    # CONFIGURATION SECTION - Edit these values to customize the execution
    # =====================================================================
    n_images = 3000          # Number of images to process (None for all)
    save_debug = False       # Whether to save debug images
    clean_output = False     # Whether to clean output directory before starting
    model_type = 'cnn'       # Model type to train: 'cnn' or 'rf'
    gpu_id = 2               # GPU ID to use (0, 1, or 2) for CNN
    
    # CNN parameters
    batch_size = 32          # Batch size for CNN training
    num_epochs = 20          # Number of epochs for CNN training
    learning_rate = 0.001    # Learning rate for optimizer
    cache_images = True      # Whether to cache images in memory
    
    # Random Forest parameters
    tune_rf = False          # Whether to tune RF hyperparameters
    rf_params = {            # Parameters for RF (used if tune_rf is False)
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'criterion': 'gini'
    }
    

    # Use project directory structure to locate files automatically
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define paths based on project structure
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_folder = os.path.join(data_dir, 'skin_type_analysis')
    
    # Create model-specific output folder
    if model_type.lower() == 'cnn':
        model_output_folder = os.path.join(project_root, 'trained_model', 'skin_type_classifier', 'cnn_model')
    else:  # Random Forest
        model_output_folder = os.path.join(project_root, 'trained_model', 'skin_type_classifier', 'rf_model')
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_output_folder, exist_ok=True)
    
    print(f"Using paths:")
    print(f"  Project root: {project_root}")
    print(f"  CSV: {csv_path}")
    print(f"  Images: {image_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Model output: {model_output_folder}")
    print(f"  Number of images: {n_images}")
    print(f"  Model type: {model_type}")
    if model_type.lower() == 'cnn':
        print(f"  Using GPU: {gpu_id}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
    else:
        print(f"  Tune RF: {tune_rf}")
        if not tune_rf:
            print(f"  RF parameters: {rf_params}")
    
    # Run the main function
    main(
        csv_path=csv_path,
        image_folder=image_folder,
        output_folder=output_folder,
        model_output_folder=model_output_folder,
        n_images=n_images,
        save_debug=save_debug,
        clean_output=clean_output,
        model_type=model_type,
        gpu_id=gpu_id,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cache_images=cache_images,
        tune_rf=tune_rf,
        rf_params=rf_params
    )