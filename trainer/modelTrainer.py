import os
import datetime
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
import tqdm
import ignite



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

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    return device


class ModelTrainer:
    def __init__(self, training_data, validation_data, test_data, training_batch_size, num_workers, shuffle,
                 training_checkpoint_data_count, validation_checkpoint_data_count, loss_fn,
                 epochs_to_train, optimizer=None, model=None, start_epoch=1):
        
        self.transform = get_transforms()
    
        self.device = get_device()

        if model is None:
            self.model = model
        else:
            self.model = model.to(self.device)

        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model, optimizer):
        self.set_model(model)
        self.set_optimizer(optimizer)

    def create_model_state_dict(self, epoch, train_loss, val_loss):
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }

        return model_state

    def save_model(self, train_loss, val_loss, epoch, best):
        model_state = self.create_model_state_dict(epoch, train_loss, val_loss)
        save_path = os.path.join(os.path.dirname(os.getcwd()), "trained_model")

        torch.save(model_state, os.path.join(save_path, ""))
        if best:
            torch.save(model_state, os.path.join(save_path, ""))

    def train_epoch(self, dataloader_index):
        self.model.train()
        train_loss = []
        loop = tqdm(self.training_dataloader[dataloader_index], leave=False)
        for image_batch, labels in loop:
            image_batch = image_batch.to(self.device)
            labels = labels.to(self.device)

            predicted_data = self.resnet(image_batch)

            loss = self.loss_fn(predicted_data, labels.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.item())
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    
    def train(self):
        if self.model is None or self.optimizer is None:
            print("Model and/or Optimizer not initialized")
            return
        
        print("Starting Training...")

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs_to_train):
            print(f"Epoch {epoch}:")

        
        print("Finished training")

    