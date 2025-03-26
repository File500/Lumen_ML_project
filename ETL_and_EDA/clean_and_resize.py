import os
import sys
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm


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


def resize_images(folder_path, output_folder=None, target_size=(224, 224)):
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

    # Process each file with progress bar
    for filename in tqdm(image_files, desc="Processing images", unit="image"):
        file_path = os.path.join(folder_path, filename)

        try:
            # Open the image
            with Image.open(file_path) as img:
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

                # Save the resulting image
                output_path = os.path.join(output_folder, filename)
                new_img.save(output_path, quality=95)

                processed += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped += 1

    print(f"\nResizing complete!")
    print(f"Processed {processed} images")
    print(f"Skipped {skipped} files")



if __name__ == "__main__":

    # # Check if command line arguments are provided
    # if len(sys.argv) < 2:
    #     print("Usage:")
    #     print("python resize_images.py <input_folder> [output_folder]")
    #     sys.exit(1)
    #
    # # Get folder path from command line arguments
    # input_folder = sys.argv[1]
    #
    # # Get output folder if provided
    # output_folder = sys.argv[2] if len(sys.argv) > 2 else None

    # test_folder_path = "./test_folder/"
    # test_output_folder_path = "./test_output_folder/"

    input_folder = '../../Lumen_Image_Data/deduplicated_train/'
    output_folder = '../../Lumen_Image_Data/train_224X224_processed/'

    # Resize images
    resize_images(input_folder, output_folder)
    print("Done!")
