import os
import torch
import numpy as np

class Config:
    # Common directory paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    
    # Data paths
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    DATASET_PATH = os.path.join(DATA_DIR, "dataset_splits")
    CSV_FILE = os.path.join(DATA_DIR, "deduplicated_monk_scale_dataset_predictions.csv")
    IMG_DIR = os.path.join(DATA_DIR, "train_300X300_processed")

   
    # Model paths
    SKIN_TONE_MODEL_PATH = os.path.join(PROJECT_DIR, "trained_model", "skin_type_classifier", 
                                        "EFNet_b3_300X300_final", "final_model.pth")
    MELANOMA_MODEL_PATH = os.path.join(PROJECT_DIR, "trained_model", 
                                       "feature_extractor_melanoma_classifier_final", 
                                       "modified_melanoma_model.pth")
    
    # Training configuration
    TEST_NUM = 1
    TRAIN_MELANOMA = True
    TRAIN_COMBINED = True
    
    # Device settings
    GPU_ID = 2
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    MELANOMA_BATCH_SIZE = 30
    COMBINED_BATCH_SIZE = 16
    NUM_WORKERS = 4
    MELANOMA_EPOCHS = 1
    COMBINED_EPOCHS = 1
    MELANOMA_LR = 0.0003
    COMBINED_LR = 0.0003

