import os
import pandas as pd
import shutil
from tqdm import tqdm

def create_labeled_data(clustering_results_path, image_folder, output_csv_path, sample_size=500):
    """
    Create a labeled dataset from clustering results for model training.
    Takes a subset of clustered images to create a balanced training set.
    
    Args:
        clustering_results_path: Path to CSV with clustering results
        image_folder: Folder containing the original images
        output_csv_path: Path to save the labeled data CSV
        sample_size: Number of images to include per skin type (if available)
    """
    # Load clustering results
    df = pd.read_csv(clustering_results_path)
    print(f"Loaded clustering results with {len(df)} entries")
    
    # Create a balanced dataset with equal representation of each skin type
    labeled_data = []
    
    # For each skin type (1-10)
    for skin_type in range(1, 11):
        # Get all images classified as this skin type
        type_df = df[df['predicted_skin_type'] == skin_type]
        
        if len(type_df) == 0:
            print(f"Warning: No images classified as skin type {skin_type}")
            continue
        
        # Take a sample (or all if fewer than sample_size)
        if len(type_df) > sample_size:
            sampled = type_df.sample(sample_size, random_state=42)
        else:
            sampled = type_df
        
        print(f"Selected {len(sampled)} images for skin type {skin_type}")
        
        # Add to labeled data
        for _, row in sampled.iterrows():
            image_name = row['image_name']
            
            # Find the image file
            image_path = None
            for ext in ['.jpg', '.png', '.jpeg']:
                test_path = os.path.join(image_folder, f"{image_name}{ext}")
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            
            if image_path:
                labeled_data.append({
                    'image_name': image_name,
                    'image_path': image_path,
                    'monk_skin_type': int(row['predicted_skin_type'])
                })
            else:
                print(f"Warning: Image {image_name} not found.")
    
    # Create DataFrame
    labeled_df = pd.DataFrame(labeled_data)
    
    # Save to CSV
    labeled_df.to_csv(output_csv_path, index=False)
    print(f"Created labeled dataset with {len(labeled_df)} images")
    print(f"Saved to {output_csv_path}")
    
    # Print distribution
    print("\nDistribution of skin types in the created dataset:")
    print(labeled_df['monk_skin_type'].value_counts().sort_index())

def main():
    """Main function to run the script."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    data_dir = os.path.join(project_root, 'data')
    clustering_results_path = os.path.join(data_dir, 'skin_type_analysis', 'isic2020_with_monk_skin_types.csv')
    image_folder = os.path.join(data_dir, 'train_224X224')
    output_csv_path = os.path.join(data_dir, 'labeled_skin_types.csv')
    
    # Check if clustering results exist
    if not os.path.exists(clustering_results_path):
        print(f"Error: Clustering results not found at {clustering_results_path}")
        print("Please run the clustering script first to generate initial classifications.")
        return
    
    # Create labeled dataset
    create_labeled_data(clustering_results_path, image_folder, output_csv_path)
    
    print("\nNext steps:")
    print("1. Run skin_tone_training.py to train a model using this labeled data")
    print("2. Or run skin_tone_main.py which will automatically detect and use this labeled data")

if __name__ == "__main__":
    main()