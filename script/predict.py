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
from tqdm import tqdm

import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pretrained_model import PretrainedMelanomaClassifier

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def mask_dark_pixels_torch(img, threshold=30, inpaint_radius=25):
    """
    Clean dark pixels using PyTorch with GPU acceleration

    Parameters:
        img: PIL.Image - Input image
        threshold: int - Darkness threshold (0-255)
        inpaint_radius: int - Radius for inpainting

    Returns:
        PIL.Image - Cleaned image
    """
    # Convert PIL image to torch tensor
    if hasattr(img, 'convert'):  # Check if it's a PIL Image
        img = img.convert('RGB')

    # Convert to tensor (0-1 range) and move to GPU
    transform = T.Compose([
        T.ToTensor(),
    ])
    img_tensor = transform(img).to(device)

    # Create grayscale version
    gray_tensor = TF.rgb_to_grayscale(img_tensor)

    # Create mask - detect pixels darker than threshold
    normalized_threshold = threshold / 255.0
    dark_mask = (gray_tensor < normalized_threshold).float()

    # Edge detection using Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # Apply Sobel filters
    gray_padded = F.pad(gray_tensor.unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    edge_x = F.conv2d(gray_padded, sobel_x)
    edge_y = F.conv2d(gray_padded, sobel_y)
    edges = torch.sqrt(edge_x ** 2 + edge_y ** 2).squeeze(0)

    # Threshold edges
    edge_mask = (edges > 0.1).float()

    # Combine masks
    combined_mask = torch.clamp(dark_mask + edge_mask, 0, 1)

    # Dilate mask for better coverage
    dilate_kernel = torch.ones(5, 5, device=device)
    dilate_kernel = dilate_kernel.view(1, 1, 5, 5)
    combined_mask_expanded = combined_mask.unsqueeze(0)
    dilated_mask = F.conv2d(
        combined_mask_expanded,
        dilate_kernel,
        padding=2
    ).squeeze(0)
    dilated_mask = (dilated_mask > 0).float()

    # Clean up small regions (approximate connected components filtering)
    # First dilate then erode (closing operation)
    close_kernel = torch.ones(7, 7, device=device)
    close_kernel = close_kernel.view(1, 1, 7, 7)
    closing_mask = F.conv2d(
        dilated_mask.unsqueeze(0),
        close_kernel,
        padding=3
    )
    closing_mask = F.conv_transpose2d(
        (closing_mask > 0).float(),
        close_kernel,
        padding=3
    ).squeeze(0)
    closing_mask = (closing_mask > 30).float()  # Threshold to remove small areas

    # Inpainting by using a weighted average of surrounding pixels
    # This is a simplified inpainting approach
    mask_3d = closing_mask.expand_as(img_tensor)

    # Create versions of the image with different blur amounts
    blur_small = TF.gaussian_blur(img_tensor, kernel_size=[5, 5], sigma=[2.0, 2.0])
    blur_medium = TF.gaussian_blur(img_tensor, kernel_size=[9, 9], sigma=[4.0, 4.0])
    blur_large = TF.gaussian_blur(img_tensor, kernel_size=[15, 15], sigma=[8.0, 8.0])

    # Blend original with increasingly blurred versions based on mask and distance
    inpainted = img_tensor * (1 - mask_3d) + blur_medium * mask_3d

    # Convert back to PIL
    result = TF.to_pil_image(inpainted.cpu())
    return result


def resize_images_torch(image, target_size):
    img_copy = image.copy()

    # Clean the image using PyTorch
    cleaned_img = mask_dark_pixels_torch(img_copy, threshold=70)

    # Create a transform for resizing, centering, and padding
    transform = T.Compose([
        T.Resize(min(target_size)),
        T.CenterCrop(min(cleaned_img.width, cleaned_img.height)),
        T.Pad(
            padding=[(target_size[0] - cleaned_img.width) // 2 if cleaned_img.width < target_size[0] else 0,
                     (target_size[1] - cleaned_img.height) // 2 if cleaned_img.height < target_size[
                         1] else 0],
            fill=0
        ),
        T.Resize(target_size)
    ])

    # Apply transforms
    final_img = transform(cleaned_img)
    return final_img


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
    print(f"Cleaning and resizing images to {resize_size[0]}x{resize_size[1]} pixels and making predictions...")
    print(f"Using device: {device}")
    binary_mode = config_dict.get("BINARY_MODE", True)
    num_classes = 2  # Keep as 2 for the model definition

    model = PretrainedMelanomaClassifier(num_classes=num_classes, binary_mode=binary_mode)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    with torch.no_grad():
        for jpg_file in tqdm(jpg_files):
            try:

                current_image = Image.open(jpg_file).convert("RGB")
                cleaned_image = resize_images_torch(current_image, resize_size)

                if test_data.size != 0:
                    image_metadata = test_data.loc[test_data.image == jpg_file.stem]

                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                composed_image_tensor = transform(cleaned_image).unsqueeze(0).to(device)

                
                predicted_data = model(composed_image_tensor)

                # Handle predictions based on model output
                if len(predicted_data.shape) == 1 or predicted_data.shape[1] == 1:  # Binary with single output
                    preds = torch.sigmoid(predicted_data)
                    preds = (preds > 0.6).float().cpu().numpy()
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
    test_data = pd.DataFrame()

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == '.csv':
            csv_file = file
            break

    if csv_file is None:
        print("No CSV file found in the folder")
    else:
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
