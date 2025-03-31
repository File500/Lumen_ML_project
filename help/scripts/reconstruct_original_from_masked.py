import cv2
import numpy as np
import os
import argparse


def reconstruct_image(original_path, masked_path, inverted_mask_path, output_path, dilation_size=2):
    """
    Reconstruct the original image by replacing the masked (black) parts of the masked image
    with the corresponding parts from the original image. Includes dilation to remove outline.

    Args:
        original_path: Path to the original unmodified image
        masked_path: Path to the masked image (with black areas)
        inverted_mask_path: Path to the inverted mask (black areas from masked image are white)
        output_path: Path to save the reconstructed image
        dilation_size: Size of dilation kernel to expand the mask (removes thin outlines)
    """
    # Read the images
    original = cv2.imread(original_path)
    masked = cv2.imread(masked_path)
    mask = cv2.imread(inverted_mask_path)

    # Check if all images were loaded successfully
    if original is None or masked is None or mask is None:
        print(f"Error loading one or more images.")
        if original is None:
            print(f"Failed to load original image: {original_path}")
        if masked is None:
            print(f"Failed to load masked image: {masked_path}")
        if mask is None:
            print(f"Failed to load mask image: {inverted_mask_path}")
        return False

    # Convert the mask to binary (white areas become 1, black areas become 0)
    # Use a threshold to handle any slight variations in the black color
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_mask, 128, 255, cv2.THRESH_BINARY)

    # Dilate the mask to expand white areas slightly (removes thin outlines)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Convert the dilated binary mask to boolean (True where white, False where black)
    bool_mask = dilated_mask > 0

    # Create a copy of the masked image
    result = masked.copy()

    # Replace the masked parts (where the mask is white) with the original
    result[bool_mask] = original[bool_mask]

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Reconstructed image saved to {output_path}")
    return True


def process_directory(originals_dir, masked_dir, mask_dir, output_dir, dilation_size=2):
    """
    Process all matching images in the directories

    Args:
        originals_dir: Directory containing original images
        masked_dir: Directory containing masked images
        mask_dir: Directory containing inverted mask images
        output_dir: Directory to save reconstructed images
        dilation_size: Size of dilation kernel to expand the mask (removes thin outlines)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the originals directory
    files = os.listdir(originals_dir)

    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Process each image file
    count = 0
    for filename in files:
        # Check if the file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            original_path = os.path.join(originals_dir, filename)
            masked_path = os.path.join(masked_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Check if all required files exist
            if os.path.exists(masked_path) and os.path.exists(mask_path):
                if reconstruct_image(original_path, masked_path, mask_path, output_path, dilation_size):
                    count += 1
            else:
                if not os.path.exists(masked_path):
                    print(f"Warning: Masked image {masked_path} does not exist")
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask image {mask_path} does not exist")

    print(f"Reconstructed {count} images from {originals_dir} to {output_dir}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct original images by replacing masked areas")
    parser.add_argument("--dilation", type=int, default=0, help="Dilation size to remove outlines (default: 2)")
    args = parser.parse_args()

    originals_dir = "../../data/train_224X224_processed"
    masked_dir = "../../data/dataset_preprocessed"
    mask_dir = "../../data/invert_masked_224X224"
    output_dir = "../../data/reconstructed_224X224"

    process_directory(originals_dir, masked_dir, mask_dir, output_dir, args.dilation)
