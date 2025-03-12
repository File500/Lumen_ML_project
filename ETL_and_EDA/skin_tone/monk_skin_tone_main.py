import os
import sys
import pandas as pd

def main():
    """
    Main script to run skin tone classification using either:
    1. Pre-trained model
    2. Training a new model from labeled data
    3. Clustering approach (fallback)
    
    This script serves as the entry point and selects the appropriate method.
    """
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add the project directory to the Python path
    sys.path.append(project_root)
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_folder = os.path.join(data_dir, 'skin_type_analysis')
    model_folder = os.path.join(project_root, 'trained_model')
    
    # Create necessary directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)
    
    # Path to pre-trained model
    pretrained_model_path = os.path.join(model_folder, 'monk_skin_tone_model.pth')
    
    # Path to training data with labeled Monk skin types (if available)
    training_data_path = os.path.join(data_dir, '')
    
    # Check which approach to use
    if os.path.exists(pretrained_model_path):
        print("Found pre-trained model. Using model-based classification...")
        # Import the model-based module
        from monk_pretrained_model import load_pretrained_model, process_dataset_with_model
        
        # Load the model
        model, device = load_pretrained_model(pretrained_model_path)
        
        # Process the dataset with the model
        results_df = process_dataset_with_model(csv_path, image_folder, output_folder, model, device)
        
    elif os.path.exists(training_data_path):
        print("Found labeled training data. Training a new model...")
        # Import the training module
        from trainer.monk_skin_tone_training import train_model, predict_with_model
        
        # Train model
        model, device = train_model(training_data_path, pretrained_model_path)
        
        if model is not None:
            print("Model training complete. Using new model for classification...")
            # Import the model-based module
            from monk_pretrained_model import process_dataset_with_model
            
            # Process the dataset with the newly trained model
            results_df = process_dataset_with_model(csv_path, image_folder, output_folder, model, device)
        else:
            print("Model training failed. Falling back to clustering approach...")
            # Import the clustering module
            from monk_skin_color_clustering import process_dataset
            
            # Process the dataset using clustering
            results_df = process_dataset(csv_path, image_folder, output_folder)
    else:
        print("No pre-trained model or labeled data found. Using clustering approach...")
        # Import the clustering module
        from monk_skin_color_clustering import process_dataset
        
        # Process the dataset using clustering
        results_df = process_dataset(csv_path, image_folder, output_folder)
    
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