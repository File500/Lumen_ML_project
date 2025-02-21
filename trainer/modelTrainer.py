import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model.model import MelanomaClassifier 
import PIL
from PIL import Image


def load_model(model_path, model, optimizer):
    model_dict = torch.load(model_path)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("cuda not available!")
        return None, None, None

    model.load_state_dict(model_dict['model_state'])
    model.to(device)
    model_epoch = model_dict['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(model_dict['optimizer_state'])

    return model, optimizer, model_epoch


def create_data_loader(data, batch_size, num_workers, shuffle):
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader

def get_transforms():
    transform = transforms.Compose([
        #transforms.Resize("", PIL.Image.LANCZOS),
        #transforms.Grayscale(),
        transforms.ToTensor(), 
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    return transform


class ModelTrainer:
    def __init__(self, training_data, validation_data, test_data, training_batch_size, num_workers, shuffle,
                 training_checkpoint_data_count, validation_checkpoint_data_count, loss_fn,
                 epochs_to_train, optimizer=None, start_epoch=1):
        
        self.transform = get_transforms()
    
    def train(self):
        print("Starting Training...")


        
        print("Finished training")

    