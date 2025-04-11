import os
import json
import sys
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pretrained_model import PretrainedMelanomaClassifier
from modelTrainer import ModelTrainer


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=1, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction
        self.pos_weight = pos_weight 

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        pt = torch.exp(-BCE_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train():
    # Keep existing paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_dir)
    config_file_path = os.path.join(project_path, "config.json")
    dataset_path = os.path.join(project_path, "data")
    # dataset_training_files = os.path.join(dataset_path, "train_224X224_processed")
    dataset_training_metadata = os.path.join(dataset_path, "deduplicated_metadata.csv")

    dataset_training_files = "../../Lumen_Image_Data/train_300X300_processed"
    # dataset_training_metadata = "../../Lumen_Image_Data/deduplicated_metadata.csv"
    
    # Load configuration
    with open(config_file_path, "r") as config_file:
        config_dict = json.load(config_file)
   
    # Set the GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = config_dict.get("GPU_ID", "0")  # Default to first GPU if not specified
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Device for tensor operations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Check if the model should use binary mode
    binary_mode = config_dict.get("BINARY_MODE", True)
    num_classes = 2  # Keep as 2 for the model definition
   
    # Load the model
    model = PretrainedMelanomaClassifier(num_classes=num_classes, binary_mode=binary_mode)
    optimizer = optim.Adam(model.parameters(), lr=config_dict["LEARNING_RATE"])
    
    # Calculate class weights for imbalanced dataset
    if binary_mode:
        # Check dataset balance
        try:
            df = pd.read_csv(dataset_training_metadata)
            malignant_count = sum(df['target'] == 1)
            benign_count = len(df) - malignant_count
            
            if malignant_count > 0:
                # Calculate positive class weight (higher weight for minority class)
                # TODO: remove this commented code if focal loss is better
                # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
                pos_weight = torch.tensor([benign_count / malignant_count])
                print(f"Ppositive class weight: {pos_weight.item():.2f}")
                print(f"Malignant count: {malignant_count} --> using focal loss")
                pos_weight=pos_weight.to(device)
                loss_fn = FocalLoss(alpha=0.95, gamma=3, reduction="mean", pos_weight=pos_weight)
            else:
                print("WARNING: No malignant samples found in dataset, using unweighted loss")
                loss_fn = nn.BCEWithLogitsLoss()
        except Exception as e:
            print(f"Error calculating class weights: {e}")
            loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
   
    # Set up the trainer with train/val/test split
    trainer = ModelTrainer(
        dataset_path=dataset_path,
        csv_file=dataset_training_metadata,
        img_dir=dataset_training_files,
        training_batch_size=config_dict["TRAINING_BATCH_SIZE"],
        num_workers=config_dict["NUM_WORKERS"],
        shuffle=config_dict["SHUFFLE"],
        train_val_test_split=[0.7, 0.15, 0.15],  # 70% train, 15% val, 15% test
        loss_fn=loss_fn,
        epochs_to_train=config_dict["EPOCHS_TO_TRAIN"],
        model=model,
        optimizer=optimizer,
        binary_mode=binary_mode
    )
    
    # Train the model
    trainer.train()
   
if __name__ == "__main__":
    train()

