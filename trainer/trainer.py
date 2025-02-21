import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model.model import MelanomaClassifier
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


    #Add hyperparamethers for model training
    trainer = ModelTrainer(
        
    )

    trainer.train()
    
