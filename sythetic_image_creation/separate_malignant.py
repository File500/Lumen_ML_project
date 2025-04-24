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
import shutil
from pathlib import Path
from PIL import Image


def analyse_folder_data(jpg_files, test_data, folder_out) -> pd.DataFrame:
    # Create output folder if it doesn't exist
    output_path = Path(folder_out)
    if not output_path.exists():
        output_path.mkdir(parents=True)

    solution = pd.DataFrame()

    for jpg_file in jpg_files:
        try:
            # Open the image file
            current_image = Image.open(jpg_file)

            # Get image metadata from test_data
            image_metadata = test_data.loc[test_data.image_name == jpg_file.stem]

            # Check if any matching metadata was found
            if not image_metadata.empty:
                # Check if target value is 1
                if image_metadata['target'].values[0] == 1:
                    # Copy the image to the output folder
                    destination = output_path / jpg_file.name
                    shutil.copy2(jpg_file, destination)
                    print(f"Moved {jpg_file.name} to output folder")

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
    ## add for final solution (now the full path needs to be passed to function) -> folder_path = ./ + folder_path
    folder_path = "../../Lumen_Image_Data/train/"
    output_path = "../../Lumen_Image_Data/malignant/"

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)

    test_df, jpg_files = read_folder_data(folder_path)

    solution_data = analyse_folder_data(jpg_files, test_df, output_path)


if __name__ == "__main__":
    main()
    print("Done!")