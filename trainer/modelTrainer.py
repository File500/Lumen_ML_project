import os
import datetime
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from model.model import MelanomaClassifier 
import PIL
from PIL import Image
import tqdm
import ignite
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




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
                 epochs_to_train,  model=None, optimizer=None, start_epoch=1):
        
        # Splitting training and validation data into subsets for checkpoints
        #train_subsets = self.create_random_dataset_with_checkpoints(training_checkpoint_data_count, training_data)
        #val_subsets = self.create_random_dataset_with_checkpoints(validation_checkpoint_data_count, validation_data)

        #self.training_dataloader = self.create_dataloaders_for_subset_data(train_subsets, training_batch_size, num_workers, shuffle)
        #self.validation_dataloader = self.create_dataloaders_for_subset_data(val_subsets, 1, num_workers, shuffle)
        #self.test_dataloader = create_data_loader(test_data, 1, num_workers, shuffle)

        self.training_dataloader = create_data_loader(training_data, training_batch_size, num_workers, shuffle)
        self.validation_dataloader = create_data_loader(validation_data, training_batch_size, num_workers, shuffle)

        self.loss_fn = loss_fn
        self.epochs_to_train = epochs_to_train
        self.start_epoch = start_epoch 
        #self.transform = get_transforms()
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

    def set_model_and_optimizer(self, model, optimizer):
        self.set_model(model)
        self.set_optimizer(optimizer)

    def create_dataloaders_for_subset_data(self, subsets, batch_size, num_workers, shuffle):
        return [create_data_loader(subset, batch_size, num_workers, shuffle) for subset in subsets]
    
    def create_random_dataset_with_checkpoints(self, checkpoint_data_count, full_dataset):
        num_chunks = len(full_dataset) // checkpoint_data_count
        split_sizes = [checkpoint_data_count] * num_chunks
        remainder = len(full_dataset) % checkpoint_data_count
        if remainder > 0:
            split_sizes.append(remainder) 

        return random_split(full_dataset, split_sizes)

    def create_model_state_dict(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        model_state = {
            'time': str(datetime.datetime.now()),
            'model_state': self.model.state_dict(),
            'model_name': type(self.model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_metrics[0],
            'train_precision': train_metrics[1],
            'train_recall': train_metrics[2],
            'train_f1': train_metrics[3],
            'val_accuracy': val_metrics[0],
            'val_precision': val_metrics[1],
            'val_recall': val_metrics[2],
            'val_f1': val_metrics[3],
        }

        return model_state

    def save_model(self, train_loss, val_loss, epoch, train_metrics, val_metrics, best):
        model_state = self.create_model_state_dict(epoch, train_loss, val_loss, train_metrics, val_metrics)
        save_path = os.path.join(os.path.dirname(os.getcwd()), "trained_model")

        torch.save(model_state, os.path.join(save_path, "last-model.pt"))
        if best:
            torch.save(model_state, os.path.join(save_path, "best-model.pt"))

    def train_epoch(self, ):
        self.model.train()
        train_loss = []
        all_preds = []
        all_labels = []

        loop = tqdm(self.training_dataloader, leave=False)
        for image_batch, labels in loop:
            image_batch = image_batch.to(self.device)
            labels = labels.to(self.device)

            predicted_data = self.model(image_batch)

            loss = self.loss_fn(predicted_data, labels.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loop.set_postfix(loss=loss.item())
            train_loss.append(loss.detach().cpu().numpy())

            preds = torch.sigmoid(predicted_data).round().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return np.mean(train_loss), accuracy, precision, recall, f1
    
    def validate_epoch(self, ):
        self.model.eval()

        with torch.no_grad():
            val_losses = []
            all_preds = []
            all_labels = []

            loop = tqdm(self.validation_dataloader, leave=False)
            for image_batch, labels in loop:
                labels = labels.to(self.device)
                for windowed_batch in image_batch:
                    windowed_batch = windowed_batch.to(self.device)

                    predicted_data = self.model(windowed_batch)
                    loss = self.loss_fn(predicted_data, labels.float())
                    val_losses.append(loss.detach().cpu().numpy())

                    preds = torch.sigmoid(predicted_data).round().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

                    loop.set_postfix(loss=loss.item())

            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

        return np.mean(val_losses), accuracy, precision, recall, f1

    
    def train(self):
        if self.model is None or self.optimizer is None:
            print("Model and/or Optimizer not initialized")
            return
        
        print("Starting Training...")

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs_to_train):
            print(f"\nEpoch {epoch}/{self.start_epoch + self.epochs_to_train - 1}")

             # Training
            train_loss, train_acc, train_prec, train_rec, train_f1 = self.train_epoch()  

            # Validation
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate_epoch()  

            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

            # Save the model at each epoch
            train_metrics = (train_acc, train_prec, train_rec, train_f1)
            val_metrics = (val_acc, val_prec, val_rec, val_f1)
            self.save_model(train_loss, val_loss, epoch, train_metrics, val_metrics, best=(val_loss < train_loss))

        
        print("Finished training")

    