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

    # No resizing needed as images are already standardized to 224x224

    # Convert to RGB if needed
    if img1.mode != 'RGB':
        img1 = img1.convert('RGB')
    if img2.mode != 'RGB':
        img2 = img2.convert('RGB')

    # Calculate multiple perceptual hashes (more robust for medical images)
    phash1 = imagehash.phash(img1, hash_size=8)  # Power of 2 (8x8 = 64 bits)
    phash2 = imagehash.phash(img2, hash_size=8)
    phash_diff = phash1 - phash2

    # Average hash is more sensitive to color changes (important for skin conditions)
    ahash1 = imagehash.average_hash(img1, hash_size=8)  # Power of 2 (8x8 = 64 bits)
    ahash2 = imagehash.average_hash(img2, hash_size=8)
    ahash_diff = ahash1 - ahash2

    # Wavelet hash is good for capturing texture differences
    whash1 = imagehash.whash(img1, hash_size=8)  # Must be power of 2
    whash2 = imagehash.whash(img2, hash_size=8)
    whash_diff = whash1 - whash2

    # Color histogram comparison (important for skin tone/condition changes)
    # Using a more detailed histogram with more bins
    hist1 = [val for channel in img1.split() for val in channel.histogram()]
    hist2 = [val for channel in img2.split() for val in channel.histogram()]
    hist_diff = sqrt(sum((a - b) ** 2 for a, b in zip(hist1, hist2)) / len(hist1))

    # Calculate structural similarity (SSIM) for added accuracy
    # This requires numpy and scikit-image
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        img1_array = np.array(img1)
        img2_array = np.array(img2)
        ssim_value = ssim(img1_array, img2_array, channel_axis=2, data_range=255)
        ssim_diff = 1 - ssim_value  # Convert to difference (0 = identical, 1 = completely different)
    except ImportError:
        ssim_diff = None

    return {
        "perceptual_hash_diff": phash_diff,
        "average_hash_diff": ahash_diff,
        "wavelet_hash_diff": whash_diff,
        "histogram_diff": hist_diff,
        "ssim_diff": ssim_diff,
        "combined_hash_score": (phash_diff + ahash_diff + whash_diff) / 3
    }

def evaluate_skin_condition_similarity(image1_path, image2_path):
    result = compare_patient_skin_images(image1_path, image2_path)

    # Significantly stricter thresholds for standardized 224x224 images
    phash_threshold = 18
    ahash_threshold = 10
    whash_threshold = 8
    hist_threshold = 200
    combined_threshold = 8
    ssim_threshold = 0.25

    # Print detailed results
    print(f"Perceptual hash difference: {result['perceptual_hash_diff']}")
    print(f"Average hash difference: {result['average_hash_diff']}")
    print(f"Wavelet hash difference: {result['wavelet_hash_diff']}")
    print(f"Histogram difference: {result['histogram_diff']}")
    print(f"Combined hash score: {result['combined_hash_score']}")
    if result['ssim_diff'] is not None:
        print(f"SSIM difference: {result['ssim_diff']}")

    # Stricter assessment criteria with multiple conditions that must be satisfied
    if ((result['combined_hash_score'] < combined_threshold) and
            (result['perceptual_hash_diff'] < phash_threshold) and
            (result['average_hash_diff'] < ahash_threshold) and
            (result['wavelet_hash_diff'] < whash_threshold) and
            (result['histogram_diff'] < hist_threshold) and
            (result['ssim_diff'] is None or result['ssim_diff'] < ssim_threshold)):
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