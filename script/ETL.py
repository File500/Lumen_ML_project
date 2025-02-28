# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import sys
import os
import PIL
from pathlib import Path
from PIL import Image
import imagehash
from math import sqrt
import copy

import cv2
from skimage.metrics import structural_similarity as ssim
def compare_patient_skin_images(image1_path, image2_path):
    # Load the images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Standardize images to a common size for comparison
    standard_size = (640, 480)  # Adjust based on your typical image resolution

    # Create copies before resizing - thumbnail() modifies in place and returns None
    img1_resized = img1.copy()
    img2_resized = img2.copy()

    # Apply thumbnail to the copies
    img1_resized.thumbnail(standard_size)
    img2_resized.thumbnail(standard_size)

    # Convert to RGB if needed
    if img1_resized.mode != 'RGB':
        img1_resized = img1_resized.convert('RGB')
    if img2_resized.mode != 'RGB':
        img2_resized = img2_resized.convert('RGB')

    # Calculate multiple perceptual hashes (more robust for medical images)
    phash1 = imagehash.phash(img1_resized)
    phash2 = imagehash.phash(img2_resized)
    phash_diff = phash1 - phash2

    # Average hash is more sensitive to color changes (important for skin conditions)
    ahash1 = imagehash.average_hash(img1_resized, hash_size=12)  # Larger hash size for more detail
    ahash2 = imagehash.average_hash(img2_resized, hash_size=12)
    ahash_diff = ahash1 - ahash2

    # Wavelet hash is good for capturing texture differences
    whash1 = imagehash.whash(img1_resized)
    whash2 = imagehash.whash(img2_resized)
    whash_diff = whash1 - whash2

    # Color histogram comparison (important for skin tone/condition changes)
    hist1 = img1_resized.histogram()
    hist2 = img2_resized.histogram()
    hist_diff = sqrt(sum((a - b) ** 2 for a, b in zip(hist1, hist2)) / len(hist1))

    return {
        "perceptual_hash_diff": phash_diff,
        "average_hash_diff": ahash_diff,
        "wavelet_hash_diff": whash_diff,
        "histogram_diff": hist_diff,
        "combined_hash_score": (phash_diff + ahash_diff + whash_diff) / 3
    }

def evaluate_skin_condition_similarity(image1_path, image2_path):
    result = compare_patient_skin_images(image1_path, image2_path)

    # Medical image thresholds - more lenient than general photo comparison
    # because we expect some variation in conditions
    phash_threshold = 15  # More lenient for skin images
    ahash_threshold = 18  # Color-sensitive hash
    whash_threshold = 15  # Texture-sensitive hash
    hist_threshold = 400  # More lenient for lighting variations
    combined_threshold = 15

    # Print detailed results
    print(f"Perceptual hash difference: {result['perceptual_hash_diff']}")
    print(f"Average hash difference: {result['average_hash_diff']}")
    print(f"Wavelet hash difference: {result['wavelet_hash_diff']}")
    print(f"Histogram difference: {result['histogram_diff']}")
    print(f"Combined hash score: {result['combined_hash_score']}")

    # Assess condition similarity using multiple metrics
    if (result['combined_hash_score'] < combined_threshold or
            (result['perceptual_hash_diff'] < phash_threshold and
             result['wavelet_hash_diff'] < whash_threshold)):
        print("Assessment: Images show similar skin condition")
        return True
    else:
        print("Assessment: Images likely show different or changed skin condition")
        return False

duplicates_df = pd.read_csv('../data/ISIC_2020_Training_Duplicates.csv')
# duplicates_df

metadata_df = pd.read_csv('../data/ISIC_2020_Training_GroundTruth_v2.csv')
# metadata_df

metadata_df = metadata_df.loc[metadata_df.target == 0]
# metadata_df

grouped_metadata = metadata_df['patient_id'].value_counts().reset_index()
# grouped_metadata

duplicates_new_df = pd.DataFrame(columns=['image_name_1', 'image_name_2'])

non_existant_images = []

unique_ids = grouped_metadata['patient_id'].tolist()
path = '../../Lumen_Image_Data/train/'
new_rows = []

for id in unique_ids:

    patient_images = (metadata_df.where(metadata_df['patient_id'] == id)
                      .dropna(axis=0)
                      .reset_index()['image_name']
                      .tolist())

    patient_images_copy = copy.deepcopy(patient_images)

    for image1 in patient_images:

        image_1_path = path + image1 + '.jpg'
        file1 = Path(image_1_path)

        if len(patient_images_copy) == 0:
            break

        if image1 in patient_images_copy:

            patient_images_copy.remove(image1)
            patient_images_subset = copy.deepcopy(patient_images_copy)

            if not file1.exists():
                non_existant_images.append(image1)
                continue
        else:
            continue

        for image2 in patient_images_subset:

            image_2_path = path + image2 + '.jpg'
            file2 = Path(image_2_path)

            if not file2.exists():
                patient_images_copy.remove(image2)
                non_existant_images.append(image2)

            else:
                images_similar = evaluate_skin_condition_similarity(image_1_path, image_2_path)
                if images_similar:
                    patient_images_copy.remove(image2)
                    new_rows.append([image1, image2])

new_df = pd.DataFrame(new_rows, columns=duplicates_new_df.columns)
duplicates_new_df = pd.concat([duplicates_new_df, new_df], ignore_index=True)

# duplicates_new_df

df_nex = pd.DataFrame(non_existant_images, columns=['do_not_exist'])

df_nex.to_csv('nonexistent_images.csv', index=False)
duplicates_new_df.to_csv('new_training_duplicates.csv', index=False)