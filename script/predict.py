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

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pylab as plt
from torch.utils.data import DataLoader


class CNN_batch(nn.Module):

    # Contructor
    def __init__(self):
        super(CNN_batch, self).__init__()

    # Prediction
    def forward(self, x):
        return x


def analyse_folder_data(jpg_files, test_data) -> pd.DataFrame:

    solution = pd.DataFrame(columns=['image_name', 'model_prediction'])
    resize_size = (640,480)
    Model = CNN_batch()

    for jpg_file in jpg_files:
        try:

            current_image = Image.open(jpg_file)

            image_metadata = test_data.loc[test_data.image_name == jpg_file.stem]

            transform = transforms.Compose([transforms.Resize(resize_size, PIL.Image.LANCZOS), transforms.ToTensor()])

            composed_image_tensor = transform(current_image)

            model_output = Model(composed_image_tensor)
            _, yhat = torch.max(model_output.data, 1)
            # yhat = torch.tensor(mock_pred)

            solution.loc[len(solution)] = [jpg_file.stem, yhat.item()]

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
