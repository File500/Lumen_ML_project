import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model.model import MelanomaClassifier
from model.pretrained_model import PretrainedMelanomaClassifier
from modelTrainer import ModelTrainer


def train():
    project_path = os.path.dirname(os.getcwd())
    config_file_path = os.path.join(project_path, "config.json")
    dataset_path = os.path.join(project_path, "data")
    dataset_training_files = os.path.join(dataset_path, "ISIC_2020_Training_JPEG")
    dataset_training_metadata = os.path.join(dataset_path, "ISIC_2020_Training_GroundTruth_v2.csv")
    dataset_testing_files = os.path.join(dataset_path, "")
    dataset_testing_metadata = os.path.join(dataset_path, "ISIC_2020_Test_Metadata.csv")

    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)


    # Load the model
    model = PretrainedMelanomaClassifier(num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=config_dict["LEARNING_RATE"])

    #Add hyperparamethers for model training
    trainer = ModelTrainer(
        training_data= "",
        validation_data="",
        test_data="",
        training_batch_size=config_dict["TRAINING_BATCH_SIZE"],
        num_workers=config_dict["NUM_WORKERS"],
        shuffle=config_dict["SHUFFLE"],
        training_checkpoint_data_count=config_dict["TRAINING_CHECKPOINT_DATA_COUNT"],
        validation_checkpoint_data_count=config_dict["VALIDATION_SPLIT_DATA_COUNT"],
        loss_fn=nn.BCEWithLogitsLoss(),
        epochs_to_train=config_dict["EPOCHS_TO_TRAIN"],
        model = model,
        optimizer = optimizer
    )

    
    trainer.train()
    
