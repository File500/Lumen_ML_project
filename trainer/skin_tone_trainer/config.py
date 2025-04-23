import os
import torch
import numpy as np

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class BaseConfig:
    # Common data paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    CSV_PATH = os.path.join(DATA_DIR, "monk_scale_dataset.csv")
    
    # Common settings
    NUM_CLASSES = 7
    FREEZE_FEATURES = True
    
    # Data split
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    
    # Device settings
    GPU_ID = 2
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    # Common training parameters
    BATCH_SIZE = 48
    TEST_BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-6  # Reduced for fine-tuning
    WEIGHT_DECAY = 1e-5 
    PATIENCE = 10

    # Transformation parameters
    RANDOMROTATION = 10
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    SATURATION = 0.2
    HUE = 0.1
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    USE_LAB = True

    TEST_NUM = 1

    MODEL_TYPE = 'b3'
    IMAGE_DIMENSION = 300
    
    # Specific directories for B3
    IMAGE_DIR = os.path.join(DATA_DIR, f"train_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_processed")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "trained_model", f"skin_type_classifier", 
                             f"EFNet_{MODEL_TYPE}_{IMAGE_DIMENSION}X{IMAGE_DIMENSION}_test{TEST_NUM}")


