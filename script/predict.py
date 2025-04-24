# Suppress warnings:
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys
import os
import PIL
from pathlib import Path
from PIL import Image
import cv2
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets as dset
import torchvision.models as models
import matplotlib.pylab as plt
from torch.utils.data import DataLoader, Dataset
import json
from tqdm import tqdm
import torch.nn.functional as F
import multiprocessing as mp

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add parent directory to path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pretrained_model import PretrainedMelanomaClassifier


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
    transform = transforms.Compose([
        transforms.ToTensor(),
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


class MelanomaDataset(Dataset):
    """Dataset class for melanoma images with image cleaning"""

    def __init__(self, img_files, resize_size=(300, 300), clean_images=True,
                 cleaning_threshold=30, inpaint_radius=25):
        self.img_files = img_files
        self.resize_size = resize_size
        self.clean_images = clean_images
        self.cleaning_threshold = cleaning_threshold
        self.inpaint_radius = inpaint_radius

        #Define normalization as the final step
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            # Clean image if enabled
            if self.clean_images:
                img = mask_dark_pixels_torch(
                    img,
                    threshold=self.cleaning_threshold,
                    inpaint_radius=self.inpaint_radius
                )

            # Resize image
            img = transforms.Resize(self.resize_size)(img)

            # Convert to tensor
            img_tensor = transforms.ToTensor()(img)

            # Normalize
            img_tensor = self.normalize(img_tensor)

            return img_tensor, img_path.stem

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return a placeholder tensor and the filename
            return torch.zeros((3, self.resize_size[0], self.resize_size[1])), img_path.stem


def analyse_folder_data_batch(jpg_files, test_data, batch_size=32, num_workers=4,
                              clean_images=True, cleaning_threshold=30) -> pd.DataFrame:
    """Analyze images in a folder and make predictions using batch processing"""
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_path = current_dir.parent

    solution = pd.DataFrame(columns=['image_name', 'model_prediction'])
    model_path = project_path / "trained_model" / "melanoma_classifier" / "best-model.pth"

    resize_size = (300,300)
    binary_mode = True
    num_classes = 2

    # Load model
    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return solution

    model = PretrainedMelanomaClassifier(num_classes=num_classes, binary_mode=binary_mode)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    clean_images = True
    #print(f"Creating dataset with image cleaning={clean_images}, threshold={cleaning_threshold}")

    # Create dataset and dataloader
    dataset = MelanomaDataset(
        jpg_files,
        resize_size=resize_size,
        clean_images=clean_images,
        cleaning_threshold=cleaning_threshold
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Disable gradient computation for the entire process
    torch.set_grad_enabled(False)

    results = []

    # Process batches
    print(f"Processing {len(jpg_files)} images in batches of {batch_size}...")

    with torch.no_grad():
        for batch_images, batch_filenames in tqdm(dataloader):
            # Move batch to device
            batch_images = batch_images.to(device)

            # Get predictions
            outputs = model(batch_images)

            # Handle predictions based on model output
            if len(outputs.shape) == 1 or outputs.shape[1] == 1:  # Binary with single output
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.6).float().cpu().numpy()
            else:  # Multi-class with multiple outputs
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            # Collect results
            for i, filename in enumerate(batch_filenames):
                results.append({
                    'image_name': filename,
                    'model_prediction': int(preds[i])
                })

    # Convert results to DataFrame
    solution = pd.DataFrame(results)

    # Print summary
    malignant_count = sum(solution['model_prediction'] == 1)
    benign_count = sum(solution['model_prediction'] == 0)
    print("\n===== Prediction Summary =====")
    print(f"Total images processed: {len(solution)}")
    print(f"Benign predictions (0): {benign_count} ({benign_count / len(solution) * 100:.1f}%)")
    print(f"Malignant predictions (1): {malignant_count} ({malignant_count / len(solution) * 100:.1f}%)")

    return solution


def read_folder_data(folder):
    """Read image files and optional CSV data from a folder"""
    folder = Path(folder)
    csv_file = None
    test_data = pd.DataFrame()

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == '.csv':
            csv_file = file
            break

    if csv_file is None:
        print("")
    else:
        try:
            test_data = pd.read_csv(csv_file)
            print(f"Loaded CSV file: {csv_file.name}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return test_data, []

    jpg_files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']])
    print(f"Found {len(jpg_files)} images in {folder}")

    return test_data, jpg_files


def main():
    """Main function to run predictions with argument parsing"""
    parser = argparse.ArgumentParser(description="Process test images for prediction.")
    parser.add_argument("--local", action="store_true", help="Use predefined local test images.")
    parser.add_argument("--uploaded", action="store_true", help="Use uploaded image folder.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for data loading")
    parser.add_argument("--clean_images", action="store_true", help="Enable image cleaning")
    parser.add_argument("--cleaning_threshold", type=int, default=30, help="Darkness threshold for cleaning (0-255)")
    args = parser.parse_args()

    # Get current directory paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_path = current_dir.parent
    data_path = project_path / "data"
    output_csv = data_path / "Test_predictions.csv"
    # Determine folder path based on arguments
    if args.local:
        folder_path = data_path / "ISIC_2020_Test_Input"
    elif args.uploaded:
        folder_path = data_path / "uploaded_images"

    else:
        print("Error: You must specify either --local or --uploaded")
        sys.exit(1)

    csv_filename_output = output_csv

    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)

    test_df, jpg_files = read_folder_data(folder_path)

    # Use the optimized batch processing function
    solution_data = analyse_folder_data_batch(
        jpg_files,
        test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clean_images=args.clean_images,
        cleaning_threshold=args.cleaning_threshold
    )

    # Save results
    output_path = Path(csv_filename_output)
    solution_data.to_csv(path_or_buf=output_path, index=False)
    print(f"Saved solutions to {output_path}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
    print("Done!")