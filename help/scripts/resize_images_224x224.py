import os
import sys
from PIL import Image

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

    # Counter for processed images
    processed = 0
    skipped = 0

    print(f"Resizing images to {target_size[0]}x{target_size[1]} pixels...")

    # Process each file
    for filename in files:
        file_path = os.path.join(folder_path, filename)

        # Skip directories
        if os.path.isdir(file_path):
            continue

        # Check if file is an image by extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in supported_formats:
            skipped += 1
            continue

        try:
            # Open the image
            with Image.open(file_path) as img:
                # Create a copy of the image to preserve the original
                img_copy = img.copy()

                # Use thumbnail function to resize while preserving aspect ratio
                img_copy.thumbnail(target_size, Image.LANCZOS)

                # Create a new image with target dimensions and paste the thumbnailed image
                new_img = Image.new("RGB", target_size, color=(0, 0, 0))

                # Calculate position to paste (center the image)
                paste_x = (target_size[0] - img_copy.width) // 2
                paste_y = (target_size[1] - img_copy.height) // 2

                # Paste the thumbnailed image onto the blank canvas
                new_img.paste(img_copy, (paste_x, paste_y))

                # Save the resulting image
                output_path = os.path.join(output_folder, filename)
                new_img.save(output_path, quality=95)

                processed += 1
                print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped += 1

    print(f"\nResizing complete!")
    print(f"Processed {processed} images")
    print(f"Skipped {skipped} files")


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) < 2:
        print("Usage:")
        print("python resize_images.py <input_folder> [output_folder]")
        sys.exit(1)

    # Get folder path from command line arguments
    input_folder = sys.argv[1]

    # Get output folder if provided
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None



    # input_folder = './train_224X224/';
    # output_folder = './train_224X224_resized/';


    # Resize images
    resize_images(input_folder, output_folder)