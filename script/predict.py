import pandas as pd
import sys
import os
from pathlib import Path
from PIL import Image


def analyse_folder_data(jpg_files, test_data) -> pd.DataFrame:
    solution = pd.DataFrame(columns=['image_name', 'model_prediction'])
    ### temp. just for script test
    counter = 10
    ###

    for jpg_file in jpg_files:
        try:

            current_image = Image.open(jpg_file)
            print(f"Processing image: {jpg_file.name}")
            print(f"Image size: {current_image.size}")

            image_metadata = test_data.loc[test_data.image_name == jpg_file.stem]
            print(f"Image metadata: {image_metadata}")

        except Exception as e:
            print(f"Error processing {jpg_file.name}: {e}")
            continue

        ### temp. just for script test
        counter -= 1
        if counter == 0:
            break
        ###

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
