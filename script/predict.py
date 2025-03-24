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
import cv2

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.models as models
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pretrained_model import PretrainedMelanomaClassifier


def mask_dark_pixels(img, threshold=30, inpaint_radius=25):
    # Convert PIL image to OpenCV format if needed
    if hasattr(img, 'convert'):  # Check if it's a PIL Image
        img = np.array(img.convert('RGB'))
        img = img[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create mask - detect pixels darker than threshold
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    # Enhance dark line detection with edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Use Hough Line Transform to detect straight lines
    lines_mask = np.zeros_like(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Draw detected lines on the lines mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness=3)

    # Combine with previous mask
    mask = cv2.bitwise_or(mask, lines_mask)

    # Dilate edges for better coverage
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.bitwise_or(mask, dilated_edges)

    # Clean up mask - remove small noise and enhance coherent lines
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    # Opening (erosion followed by dilation) - removes small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Closing (dilation followed by erosion) - closes small gaps in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Optional: Use connected components to filter out small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:  # Filter regions smaller than 100 pixels
            mask[labels == i] = 0

    # Apply mask using inpainting
    result = cv2.inpaint(img_rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)

    # Return only the cleaned image
    return result


def resize_images(img, target_size=(224, 224)):

    print(f"Cleaning and resizing image to {target_size[0]}x{target_size[1]} pixels...")
    # Create a copy of the image to preserve the original
    img_copy = img.copy()

    # Clean the image using mask_dark_pixels function
    cleaned_img = mask_dark_pixels(img_copy, threshold=70)

    # Convert back to PIL image from NumPy array
    cleaned_pil = Image.fromarray(cleaned_img)

    # Use thumbnail function to resize while preserving aspect ratio
    cleaned_pil.thumbnail(target_size, Image.LANCZOS)

    # Create a new image with target dimensions and paste the thumbnailed image
    new_img = Image.new("RGB", target_size, color=(0, 0, 0))

    # Calculate position to paste (center the image)
    paste_x = (target_size[0] - cleaned_pil.width) // 2
    paste_y = (target_size[1] - cleaned_pil.height) // 2

    # Paste the thumbnailed image onto the blank canvas
    new_img.paste(cleaned_pil, (paste_x, paste_y))
    print("Done processing image...")
    return new_img


def analyse_folder_data(jpg_files, test_data) -> pd.DataFrame:
    device = torch.device("cpu")

    solution = pd.DataFrame(columns=['image_name', 'model_prediction'])
    model_path = "../trained_model/melanoma_classifier/best-model.pth"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_dir)
    config_file_path = os.path.join(project_path, "config.json")

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)

    resize_size = (config_dict.get("RESIZE_WIDTH"), config_dict.get("RESIZE_HIGHT"))
    binary_mode = config_dict.get("BINARY_MODE", True)
    num_classes = 2  # Keep as 2 for the model definition

    model = PretrainedMelanomaClassifier(num_classes=num_classes, binary_mode=binary_mode)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    for jpg_file in jpg_files:
        try:

            current_image = Image.open(jpg_file).convert("RGB")
            cleaned_image = resize_images(current_image, resize_size)

            image_metadata = test_data.loc[test_data.image == jpg_file.stem]

            transform = transforms.Compose([transforms.Resize(resize_size),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])

            composed_image_tensor = transform(cleaned_image).unsqueeze(0).to(device)

            predicted_data = model(composed_image_tensor)

            # Handle predictions based on model output
            if len(predicted_data.shape) == 1 or predicted_data.shape[1] == 1:  # Binary with single output
                preds = torch.sigmoid(predicted_data).round().detach().numpy()
            else:  # Multi-class with multiple outputs
                preds = torch.argmax(predicted_data, dim=1, keepdim=True).detach().numpy()

            # print(int(preds[0, 0]))
            solution.loc[len(solution)] = [jpg_file.stem, int(preds[0, 0])]

        except Exception as e:
            print(f"Error processing {jpg_file.name}: {e}")
            continue

    return solution


def read_folder_data(folder):
    folder = Path(folder)
    csv_file = None

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == '.csv':
            csv_file = file
            break

    if csv_file is None:
        print("No CSV file found in the folder")
        return

    try:
        test_data = pd.read_csv(csv_file)
        print(f"Loaded CSV file: {csv_file.name}")

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    jpg_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']])

    return test_data, jpg_files


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <folder_path> <csv_filename>")
        sys.exit(1)

    ## add for final solution (now the full path needs to be passed to function) -> folder_path = ./ + folder_path 
    folder_path = sys.argv[1]
    csv_filename_output = sys.argv[2]

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)

    test_df, jpg_files = read_folder_data(folder_path)

    solution_data = analyse_folder_data(jpg_files, test_df)

    solution_data.to_csv(path_or_buf=f"./{csv_filename_output}", index=False)

    print(f"Saved solutions to {csv_filename_output}")


if __name__ == "__main__":
    main()
    print("Done!")
