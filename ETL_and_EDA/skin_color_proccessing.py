import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import shutil

def extract_skin_color(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reshape the image
    pixels = img.reshape((-1, 3))
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    
    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def classify_skin_type(dominant_color):
    # Define color ranges for Fitzpatrick types
    fitzpatrick_ranges = {
        'I': np.array([255, 236, 210]),
        'II': np.array([255, 218, 184]),
        'III': np.array([228, 185, 142]),
        'IV': np.array([198, 134, 66]),
        'V': np.array([141, 85, 36]),
        'VI': np.array([84, 57, 33])
    }
    
    # Calculate distances to each Fitzpatrick type
    distances = {skin_type: np.linalg.norm(dominant_color - color) 
                 for skin_type, color in fitzpatrick_ranges.items()}
    
    # Classify based on the smallest distance
    classified_type = min(distances, key=distances.get)
    return classified_type

def process_images(input_folder, output_folder):
    # Create output folders for each skin type
    for skin_type in ['I', 'II', 'III', 'IV', 'V', 'VI']:
        os.makedirs(os.path.join(output_folder, skin_type), exist_ok=True)
    
    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            dominant_color = extract_skin_color(image_path)
            skin_type = classify_skin_type(dominant_color)
            
            # Move the image to the corresponding output folder
            destination = os.path.join(output_folder, skin_type, filename)
            shutil.copy(image_path, destination)
            print(f"Classified {filename} as Fitzpatrick Type {skin_type}")

# Usage
input_folder = 'path/to/input/images'
output_folder = 'path/to/output/folders'
process_images(input_folder, output_folder)
