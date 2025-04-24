import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import the configuration and pipeline
from config import Config
from pipeline_trainer import run_transfer_learning_pipeline

def main():
    """
    Main entry point for running the melanoma classification pipeline.
    """
    try:
        # Convert Config class to dictionary
        config_dict = {
            "dataset_path": Config.DATASET_PATH,
            "csv_file": Config.CSV_FILE,
            "img_dir": Config.IMG_DIR,
            "skin_model_path": Config.SKIN_TONE_MODEL_PATH,
            "melanoma_model_path": Config.MELANOMA_MODEL_PATH,
            "train_melanoma": Config.TRAIN_MELANOMA,
            "train_combined": Config.TRAIN_COMBINED,
            "melanoma_batch_size": Config.MELANOMA_BATCH_SIZE,
            "combined_batch_size": Config.COMBINED_BATCH_SIZE,
            "num_workers": Config.NUM_WORKERS,
            "melanoma_epochs": Config.MELANOMA_EPOCHS,
            "combined_epochs": Config.COMBINED_EPOCHS,
            "melanoma_lr": Config.MELANOMA_LR,
            "combined_lr": Config.COMBINED_LR,
            "device": Config.DEVICE,
            "test_num": Config.TEST_NUM
        }
        
        # Run the transfer learning pipeline
        run_transfer_learning_pipeline(config_dict)
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()