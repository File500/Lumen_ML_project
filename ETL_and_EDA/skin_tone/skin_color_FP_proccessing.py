import os
import pandas as pd
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm 

def extract_skin_color(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0]

def classify_skin_type(dominant_color):
    fitzpatrick_ranges = {
        'I': np.array([255, 236, 210]),
        'II': np.array([255, 218, 184]),
        'III': np.array([228, 185, 142]),
        'IV': np.array([198, 134, 66]),
        'V': np.array([141, 85, 36]),
        'VI': np.array([84, 57, 33])
    }
    distances = {skin_type: np.linalg.norm(dominant_color - color) 
                 for skin_type, color in fitzpatrick_ranges.items()}
    return min(distances, key=distances.get)

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define paths
data_dir = os.path.join(project_root, 'data')
csv_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth_v2.csv')
image_folder = os.path.join(data_dir, 'train_224X224')
output_csv_path = os.path.join(data_dir, 'ISIC_2020_Train_Metadata_with_skin_type.csv')

# Read the original CSV
df = pd.read_csv(csv_path)
print(f"Loaded CSV with {len(df)} entries.")

# Get list of image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]
print(f"Found {len(image_files)} images in the folder.")

# Create a dictionary to store skin types
skin_types = {}

# Process each image
print("Processing images and classifying skin types:")
for image_file in tqdm(image_files):
    image_path = os.path.join(image_folder, image_file)
    dominant_color = extract_skin_color(image_path)
    skin_type = classify_skin_type(dominant_color)
    skin_types[os.path.splitext(image_file)[0]] = skin_type
    
    # Print detailed information for every 100th image
    if len(skin_types) % 100 == 0:
        print(f"\nProcessed {len(skin_types)} images")
        print(f"Last processed image: {image_file}")
        print(f"Dominant color: {dominant_color}")
        print(f"Classified skin type: {skin_type}")

print("\nImage processing complete.")

# Add skin type column to the DataFrame
df['skin_type'] = df['image_name'].map(skin_types)

# Print skin type distribution
skin_type_counts = df['skin_type'].value_counts()
print("\nSkin type distribution:")
print(skin_type_counts)

# Save the updated DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)

print(f"\nUpdated CSV saved as {output_csv_path}")
print(f"Total images processed: {len(skin_types)}")
print(f"Total entries in CSV: {len(df)}")
