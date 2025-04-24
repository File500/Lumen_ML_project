import cv2
import numpy as np
import os

# Load an image
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

# White balance correction (Gray World)
def white_balance(img):
    img = img.astype(np.float32)

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg = (avg_b + avg_g + avg_r) / 3.0

    img[:, :, 0] = img[:, :, 0] * (avg / avg_b)
    img[:, :, 1] = img[:, :, 1] * (avg / avg_g)
    img[:, :, 2] = img[:, :, 2] * (avg / avg_r)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# Convert to Lab color space & extract skin tone channel
def convert_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    return lab, a  # 'a' contains skin tone info

# Histogram equalization for brightness correction
def histogram_equalization(img):
    lab, a = convert_lab(img)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)  # Equalize brightness
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2RGB)
    return img_eq

# Process all images in a folder
def preprocess_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        img = load_image(img_path)
        img = white_balance(img)
        img = histogram_equalization(img)

        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Save processed image

# Example usage
preprocess_images("masked_224X224/binary", "dataset_preprocessed")
