import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract

'''
../../../Lumen_Image_Data/train/ISIC_0134357.jpg
../../../Lumen_Image_Data/train/ISIC_0082348.jpg
'''
def mask_dark_pixels(image_path, output_path=None, threshold=30, inpaint_radius=25):
    # Read image
    img = cv2.imread(image_path)
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

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title("Dark Pixels Mask")
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(result)
    plt.title("Image with Dark Pixels Removed")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save result if output path is provided
    if output_path:
        Image.fromarray(result).save(output_path)

    return img_rgb, mask, result


original, mask, processed = mask_dark_pixels('../../Lumen_Image_Data/train/ISIC_0082348.jpg', threshold=70)

original, mask, processed = mask_dark_pixels('../../Lumen_Image_Data/train/ISIC_0134357.jpg', threshold=70)