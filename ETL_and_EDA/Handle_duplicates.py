import os
import shutil
import pandas as pd


def move_images_with_metadata(source_path, destination_path, duplication_list, source_csv, destination_csv):
    try:
        if not os.path.isdir(source_path):
            print(f"Error: Source folder '{source_path}' does not exist.")
            return False

        if not os.path.exists(destination_path):
            print(f"Destination folder '{destination_path}' does not exist. Creating it...")
            os.makedirs(destination_path)

        if not os.path.isfile(duplication_list):
            print(f"Error: Duplication list file '{duplication_list}' does not exist.")
            return False

        dup_df = pd.read_csv(duplication_list)
        skip_files = dup_df.iloc[:, 1].tolist()

        #Read the source CSV file that contains metadata for images
        metadata_df = pd.read_csv(source_csv)

        rows_to_copy = []

        moved_count = 0
        skipped_count = 0
        override_count = 0
        copied_rows_count = 0

        # Find the index of the 'target' column if it exists
        target_col_idx = metadata_df.columns.get_loc('target') if 'target' in metadata_df.columns else None

        for filename in os.listdir(source_path):
            source_file = os.path.join(source_path, filename)

            if not os.path.isfile(source_file):
                continue

            filename_without_ext = os.path.splitext(filename)[0]

            # Find the corresponding row in the CSV to check the target value
            matching_rows = metadata_df[metadata_df.iloc[:, 0] == filename_without_ext]

            # Skip if no match found in the metadata
            if matching_rows.empty:
                continue

            # Check if this file is in the duplication list
            is_duplicate = filename_without_ext in skip_files

            # Check if any matching row has target=1 (override condition)
            has_target_1 = False
            if target_col_idx is not None:
                has_target_1 = any(matching_rows.iloc[:, target_col_idx] == 1)

            # If it's in the duplicate list AND doesn't have target=1, skip it
            if is_duplicate and not has_target_1:
                skipped_count += 1
                continue

            # If it's in the duplicate list but HAS target=1, override the skip
            if is_duplicate and has_target_1:
                override_count += 1

            #Copy file
            destination_file = os.path.join(destination_path, filename)
            shutil.copy2(source_file, destination_file)
            moved_count += 1

            #Add rows for output CSV
            rows_to_copy.append(matching_rows)
            copied_rows_count += len(matching_rows)


        if rows_to_copy:
            new_metadata_df = pd.concat(rows_to_copy, ignore_index=True)
        else:
            new_metadata_df = pd.DataFrame(columns=metadata_df.columns)

        #Save the new metadata CSV
        new_metadata_df.to_csv(destination_csv, index=False)

        print(f"Operation completed. Files copied: {moved_count}")
        print(f"Files skipped (in duplicate list & not target=1): {skipped_count}")
        print(f"Files copied despite being duplicates (target=1): {override_count}")
        print(f"CSV rows copied: {copied_rows_count}")
        return True

    except Exception as e:
        print(f"An error occurred: {str(e)}")
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


if __name__ == "__main__":
    dataset_paths = [
        "./etl_data/new_training_duplicates.csv",
        "../data/ISIC_2020_Training_Duplicates.csv"
    ]

    combined_data = combine_image_pair_datasets(
        dataset_paths,
        output_path="combined_unique_pairs.csv"
    )

    move_images_with_metadata('../../Lumen_Image_Data/train',
               '../../Lumen_Image_Data/deduplicated_train', #to be created
               './combined_unique_pairs.csv',
               '../data/ISIC_2020_Training_GroundTruth_v2.csv',
               '../data/deduplicated_metadata.csv') #to be created