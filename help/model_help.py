import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
from pathlib import Path
import PIL

def read_folder_data(folder):
    folder = Path(folder)
    csv_file = None

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == '.csv':
            csv_file = file
            break

    if csv_file is None:
        raise Exception("No CSV file found in the folder")

    try:
        data = pd.read_csv(csv_file)
        print(f"Loaded CSV file: {csv_file.name}")
    except Exception as e:
        raise Exception(f"Error loading CSV file: {e}")

    jpg_files = sorted([f for f in folder.iterdir()
                        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg']])

    return data, jpg_files

class LargeImageDataset(Dataset):
    def __init__(self, folder_path, patch_size=640, scale_range=(0.5, 1.0), is_train=True):
        self.patch_size = patch_size
        self.scale_range = scale_range

        # Load data from folder
        self.metadata, self.image_files = read_folder_data(folder_path)

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def create_patches(self, image):
        patches = []
        w, h = image.size
        for y in range(0, h-self.patch_size+1, self.patch_size):
            for x in range(0, w-self.patch_size+1, self.patch_size):
                patch = image.crop((x, y, x+self.patch_size, y+self.patch_size))
                patches.append(patch)
        return patches

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path)

        # Get label from metadata
        label = self.metadata.loc[self.metadata.image_name == image_path.stem]['label'].iloc[0]
        label = torch.tensor(label, dtype=torch.float)

        # Random scale for multi-scale training
        scale = random.uniform(*self.scale_range)
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        image = image.resize(new_size, PIL.Image.LANCZOS)

        # Create patches
        patches = self.create_patches(image)

        # Randomly select one patch for training
        patch = random.choice(patches)

        # Transform patch
        patch_tensor = self.transform(patch)

        return patch_tensor, label

    def __len__(self):
        return len(self.image_files)

class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Binary, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)

class MobileNetV2Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Binary, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.mobilenet.last_channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mobilenet(x)

class EfficientNetB0Binary(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB0Binary, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.efficientnet(x)

def train_progressive(model, folder_path, num_epochs, patch_sizes=[320, 480, 640], device='cuda'):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for patch_size in patch_sizes:
        print(f"Training with patch size: {patch_size}")

        # Create dataset with current patch size
        dataset = LargeImageDataset(folder_path, patch_size=patch_size)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (patches, labels) in enumerate(dataloader):
                patches = patches.to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(patches)

                # Calculate loss and backward pass
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                predicted = (outputs.data > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

                running_loss += loss.item()

                if batch_idx % 100 == 99:
                    accuracy = 100 * correct / total
                    print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, '
                          f'Loss: {running_loss / 100:.3f}, '
                          f'Accuracy: {accuracy:.2f}%')
                    running_loss = 0.0
                    correct = 0
                    total = 0

        # Save checkpoint after each patch size
        torch.save({
            'patch_size': patch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_patch{patch_size}.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model (choose one)
    model = ResNet50Binary(pretrained=True)
    # model = MobileNetV2Binary(pretrained=True)
    # model = EfficientNetB0Binary(pretrained=True)

    # Folder containing images and CSV
    folder_path = "path/to/your/data/folder"

    # Train with progressive resizing
    train_progressive(model, folder_path, num_epochs=5, device=device)

if __name__ == '__main__':
    main()