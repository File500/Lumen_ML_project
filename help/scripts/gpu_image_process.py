import os
import sys
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import concurrent.futures


def check_gpu():
    """Check if CUDA is available for GPU acceleration"""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU acceleration available: {device_count} device(s)")
        print(f"Using: {device_name}")
    else:
        print("GPU acceleration not available. Using CPU.")
    return cuda_available


def mask_dark_pixels_gpu(img, threshold=30, inpaint_radius=25, use_gpu=False):
    """Enhanced version with GPU acceleration where possible"""
    # Convert PIL image to OpenCV format if needed
    if hasattr(img, 'convert'):  # Check if it's a PIL Image
        img = np.array(img.convert('RGB'))
        img = img[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV

    # Use CUDA-enabled functions if GPU is available
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # Upload image to GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # Convert to grayscale on GPU
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        gray = gpu_gray.download()

        # Threshold operation
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        # Edge detection on GPU
        gpu_edges = cv2.cuda.createCannyEdgeDetector(50, 150)
        gpu_edges_result = gpu_edges.detect(gpu_gray)
        edges = gpu_edges_result.download()
    else:
        # CPU fallback
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(gray, 50, 150)

    # Hough Line Transform (CPU-only in OpenCV)
    lines_mask = np.zeros_like(gray)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Draw detected lines on the lines mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness=3)

    # Combine with previous mask
    mask = cv2.bitwise_or(mask, lines_mask)

    # Morphological operations
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # Upload mask to GPU
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(mask)

        # GPU dilate for edges
        gpu_edges = cv2.cuda_GpuMat()
        gpu_edges.upload(edges)
        kernel3x3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gpu_dilated_edges = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_DILATE, cv2.CV_8UC1, kernel3x3).apply(gpu_edges)
        dilated_edges = gpu_dilated_edges.download()

        # Combine with mask
        mask = cv2.bitwise_or(mask, dilated_edges)

        # Upload updated mask
        gpu_mask.upload(mask)

        # Morphological operations on GPU
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        gpu_morph = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN, cv2.CV_8UC1, kernel_open)
        gpu_result = gpu_morph.apply(gpu_mask)

        gpu_morph = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, cv2.CV_8UC1, kernel_close)
        gpu_result = gpu_morph.apply(gpu_result)

        mask = gpu_result.download()
    else:
        # CPU fallback
        dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.bitwise_or(mask, dilated_edges)

        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Connected components (CPU-only in OpenCV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 100:  # Filter regions smaller than 100 pixels
            mask[labels == i] = 0

    # Ensure we're working with RGB image
    if use_gpu:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inpainting (CPU-only in OpenCV)
    result = cv2.inpaint(img_rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_NS)

    return result


def process_image(file_info, use_gpu, threshold=70, target_size=(224, 224)):
    """Process a single image with GPU acceleration"""
    file_path, output_path = file_info

    try:
        # Open the image
        with Image.open(file_path) as img:
            # Create a copy of the image to preserve the original
            img_copy = img.copy()

            # Clean the image using GPU-accelerated function
            cleaned_img = mask_dark_pixels_gpu(img_copy, threshold=threshold, use_gpu=use_gpu)

            # Convert back to PIL image from NumPy array
            cleaned_pil = Image.fromarray(cleaned_img)

            # For resizing, we can use GPU acceleration through PyTorch if available
            if use_gpu:
                # Convert to PyTorch tensor
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(max(target_size), antialias=True)
                ])

                tensor_img = transform(cleaned_pil).unsqueeze(0)

                # Move to GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tensor_img = tensor_img.to(device)

                # Convert back to PIL
                to_pil = transforms.ToPILImage()
                resized_img = to_pil(tensor_img.squeeze(0).cpu())
            else:
                # Use PIL's resize (CPU)
                cleaned_pil.thumbnail(target_size, Image.LANCZOS)
                resized_img = cleaned_pil

            # Create a new image with target dimensions and paste the thumbnailed image
            new_img = Image.new("RGB", target_size, color=(0, 0, 0))

            # Calculate position to paste (center the image)
            paste_x = (target_size[0] - resized_img.width) // 2
            paste_y = (target_size[1] - resized_img.height) // 2

            # Paste the thumbnailed image onto the blank canvas
            new_img.paste(resized_img, (paste_x, paste_y))

            # Save the resulting image
            new_img.save(output_path, quality=95)
            return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def resize_images_gpu(folder_path, output_folder=None, target_size=(224, 224), batch_size=16):
    """Process images in parallel with GPU acceleration where possible"""
    # Check for GPU availability
    use_gpu = check_gpu() and cv2.cuda.getCudaEnabledDeviceCount() > 0

    # If GPU is available, determine appropriate number of workers
    if use_gpu:
        print("Using GPU acceleration for image processing")
        # Use fewer workers when using GPU to avoid memory issues
        max_workers = min(4, os.cpu_count() or 1)
    else:
        print("GPU acceleration not available for OpenCV. Falling back to CPU")
        # Use more workers for CPU processing
        max_workers = min(16, os.cpu_count() or 1)

    print(f"Using {max_workers} worker threads")

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

    # Prepare file paths for processing
    file_infos = [(os.path.join(folder_path, filename),
                   os.path.join(output_folder, filename))
                  for filename in image_files]

    # Process files in parallel
    processed = 0
    skipped = 0

    print(f"Cleaning and resizing {len(file_infos)} images to {target_size[0]}x{target_size[1]} pixels...")

    # Process images in batches to manage memory better with GPU
    for i in range(0, len(file_infos), batch_size):
        batch = file_infos[i:i+batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a tqdm progress bar for the current batch
            results = list(tqdm(
                executor.map(lambda x: process_image(x, use_gpu, threshold=70, target_size=target_size), batch),
                total=len(batch),
                desc=f"Processing batch {i//batch_size + 1}/{(len(file_infos) + batch_size - 1)//batch_size}",
                unit="image"
            ))

            # Update counters
            processed += sum(results)
            skipped += len(results) - sum(results)

        # Clear CUDA cache between batches if using GPU
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nResizing complete!")
    print(f"Processed {processed} images")
    print(f"Skipped {skipped} files")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        # Default paths if not provided via command line
        input_folder = '../../Lumen_Image_Data/train/'
        output_folder = '../../train_224X224_processed/'
    else:
        # Get folder path from command line arguments
        input_folder = sys.argv[1]
        # Get output folder if provided
        output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    # Process images with GPU acceleration
    resize_images_gpu(input_folder, output_folder)
    print("Done!")