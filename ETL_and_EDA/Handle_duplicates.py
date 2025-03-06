import os
import shutil
import pandas as pd

def move_file(source_path, destination_path):
    try:
        if not os.path.isfile(source_path):
            print(f"Error: Source file '{source_path}' does not exist.")
            return False

        if not os.path.exists(destination_path):
            print(f"Destination folder '{destination_path}' does not exist. Creating it...")
            os.makedirs(destination_path)

        shutil.move(source_path, destination_path)
        return True

    except Exception as e:
        print(f"Error moving file: {e}")
        return False

def combine_image_pair_datasets(file_paths, output_path=None):
    all_pairs = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_pairs.append(df)
        else:
            print(f"Warning: File {file_path} not found and will be skipped.")

    if not all_pairs:
        raise ValueError("No valid datasets found to combine")

    combined_df = pd.concat(all_pairs, ignore_index=True)

    combined_df['sorted_pair'] = combined_df.apply(
        lambda row: tuple(sorted([row['image_name_1'], row['image_name_2']])),
        axis=1
    )

    deduplicated_df = combined_df.drop_duplicates(subset=['sorted_pair'])

    final_df = deduplicated_df.drop(columns=['sorted_pair'])

    if output_path:
        final_df.to_csv(output_path, index=False)
        print(f"Combined dataset saved to {output_path}")
        print(f"Original total rows: {len(combined_df)}")
        print(f"After deduplication: {len(final_df)}")

    return final_df

def main():
    dataset_paths = [
        "./etl_data/new_training_duplicates.csv",
        "../data/ISIC_2020_Training_Duplicates.csv"
    ]

    combined_data = combine_image_pair_datasets(
        dataset_paths,
        output_path="combined_unique_pairs.csv"
    )

    move_file('','') #yet to be implemented with combined data

if __name__ == "__main__":
    main()