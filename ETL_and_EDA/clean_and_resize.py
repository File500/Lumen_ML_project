import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

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


def resize_images_torch(folder_path, output_folder=None, target_size=(456, 456)):
    """
    Resize and clean images using PyTorch with GPU acceleration

    Parameters:
        folder_path: str - Path to folder with images
        output_folder: str or None - Output folder path
        target_size: tuple - Target size (width, height)
    """
    # If no output folder specified, use the input folder (overwrites original files)
    if output_folder is None:
        output_folder = folder_path
    else:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # Filter only supported image files
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_formats
                   and not os.path.isdir(os.path.join(folder_path, f))]

    # Counter for processed images
    processed = 0
    skipped = 0

    print(f"Cleaning and resizing images to {target_size[0]}x{target_size[1]} pixels...")
    print(f"Using device: {device}")

    # Process each file with progress bar
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        file_path = os.path.join(folder_path, filename)

        try:
            # Open the image
            with Image.open(file_path) as img:
                # Create a copy of the image to preserve the original
                img_copy = img.copy()

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

                # Save the resulting image
                output_path = os.path.join(output_folder, filename)
                final_img.save(output_path, quality=95)

                processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped += 1

    print(f"\nResizing complete!")
    print(f"Processed {processed} images")
    print(f"Skipped {skipped} files")


def batch_process_torch(folder_path, output_folder=None, target_size=(456, 456), batch_size=16):
    """
    Process images in batches for better GPU utilization

    Parameters:
        folder_path: str - Path to folder with images
        output_folder: str or None - Output folder path
        target_size: tuple - Target size (width, height)
        batch_size: int - Number of images to process at once
    """
    if output_folder is None:
        output_folder = folder_path
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Get image files
    files = os.listdir(folder_path)
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_formats
                   and not os.path.isdir(os.path.join(folder_path, f))]

    # Process in batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    processed = 0
    skipped = 0

    print(f"Processing {len(image_files)} images in {total_batches} batches...")
    print(f"Using device: {device}")

    for batch_idx in range(total_batches):
        batch_files = image_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_imgs = []
        batch_paths = []

        # Load batch
        for filename in batch_files:
            file_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                img = Image.open(file_path).convert('RGB')
                batch_imgs.append(img)
                batch_paths.append(output_path)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                skipped += 1

        # Process batch if any images were loaded
        if batch_imgs:
            # Process each image in the batch
            for i, (img, output_path) in enumerate(zip(batch_imgs, batch_paths)):
                if os.path.isfile(output_path):
                    skipped += 1
                    continue
                try:
                    # Clean and resize
                    cleaned_img = mask_dark_pixels_torch(img, threshold=70)

                    # Resize
                    transform = T.Compose([
                        T.Resize(min(target_size)),
                        T.Pad(
                            padding=[(target_size[0] - min(cleaned_img.width, target_size[0])) // 2,
                                     (target_size[1] - min(cleaned_img.height, target_size[1])) // 2],
                            fill=0
                        ),
                        T.Resize(target_size)
                    ])
                    final_img = transform(cleaned_img)

                    # Save result
                    final_img.save(output_path, quality=95)
                    processed += 1

                    # Free memory
                    if hasattr(img, 'close'):
                        img.close()

                except Exception as e:
                    print(f"Error processing image {i} in batch {batch_idx}: {e}")
                    skipped += 1

        # Status update
        print(f"Processed batch {batch_idx + 1}/{total_batches}, "
              f"total processed: {processed}, skipped: {skipped}")

        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nProcessing complete!")
    print(f"Processed {processed} images")
    print(f"Skipped {skipped} files")


if __name__ == "__main__":
    # Check command line arguments if provided
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default paths
        input_folder = '../../Lumen_Image_Data/deduplicated_train/'
        output_folder = '../../Lumen_Image_Data/train_456X456_processed/'

    # Print CUDA information
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available. Will use CPU processing.")

    # Choose processing method based on dataset size
    num_files = len([f for f in os.listdir(input_folder)
                     if os.path.isfile(os.path.join(input_folder, f))])

    if num_files > 100:  # For larger datasets, use batch processing
        print(f"Large dataset detected ({num_files} files). Using batch processing.")
        batch_process_torch(input_folder, output_folder, batch_size=16)
    else:
        # For smaller datasets, use single image processing
        resize_images_torch(input_folder, output_folder)

    print("Done!")