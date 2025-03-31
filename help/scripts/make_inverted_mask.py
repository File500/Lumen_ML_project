

import cv2
import numpy as np
import os
import argparse


def process_image(image_path, output_path):
    """
    Process an image to:
    1. Turn black masked areas (RGB = 0,0,0) to white (RGB = 255,255,255)
    2. Turn all non-black areas to black (RGB = 0,0,0)

    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image
    """
    # Read the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False

    # Create a mask where black pixels (0,0,0) are True
    # Using a small threshold to account for possible compression artifacts
    black_mask = np.all(img <= 5, axis=2)

    # Create a new blank (black) image
    result = np.zeros_like(img)

    # Set pixels that were black in the original to white in the result
    result[black_mask] = [255, 255, 255]

    # Save the result
    cv2.imwrite(output_path, result)
    print(f"Processed image saved to {output_path}")
    return True


def process_directory(input_dir, output_dir):
    """
    Process all images in a directory

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all files in the input directory
    files = os.listdir(input_dir)

    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Process each image file
    count = 0
    for filename in files:
        # Check if the file is an image
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if process_image(input_path, output_path):
                count += 1

    print(f"Processed {count} images from {input_dir} to {output_dir}")


if __name__ == "__main__":
    input_dir = "../../data/masked_224X224/binary"  # Change to your input directory
    output_dir = "../../data/invert_masked_224X224"
    os.makedirs(output_dir, exist_ok=True)

    process_directory(input_dir, output_dir)
    #process_image(input_dir, output_dir)


