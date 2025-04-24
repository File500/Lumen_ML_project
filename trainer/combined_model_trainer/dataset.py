import os
import torch
import collections
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CachedMelanomaDataset(Dataset):
    """
    A custom dataset for melanoma images with LRU caching.
    
    Args:
        dataframe (pd.DataFrame): DataFrame containing image info and labels
        img_dir (str): Directory with all the images
        transform (callable, optional): Optional transform to be applied on a sample
        binary_mode (bool): Whether to use binary mode (single output) or multi-class mode
        cache_images (bool): Whether to cache all images in memory
    """
    def __init__(self, dataframe, img_dir, transform=None, binary_mode=True, cache_images=True):
        self.data_frame = dataframe
        self.img_dir = img_dir
        self.binary_mode = binary_mode
        
        # Default transform if none provided
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Caching configuration
        self.cache_images = cache_images
        self.cache_size = 5000
        self.cache = collections.OrderedDict() if cache_images else {}
        
        if cache_images:
            print(f"Using LRU cache with maximum size of {self.cache_size} images")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Retrieve image from cache or load from disk
        if self.cache_images and idx in self.cache:
            # Move this item to the end (most recently used)
            image = self.cache.pop(idx)
            self.cache[idx] = image
        else:
            # Load image from disk
            img_name = self.data_frame.iloc[idx, 0]
            
            # Ensure correct file extension
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_name = img_name + '.jpg'
            
            # Create full path
            img_path = os.path.join(self.img_dir, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Add to cache if caching is enabled
                if self.cache_images:
                    # If cache is full, remove the oldest item
                    if len(self.cache) >= self.cache_size:
                        self.cache.popitem(last=False)
                    # Add current item to the end
                    self.cache[idx] = image
                    
            except FileNotFoundError:
                raise FileNotFoundError(f"Could not find image: {img_path}")
        
        # Prepare target label
        target = self.data_frame.iloc[idx, 8]  # Assuming 'target' is at column 8
        transform_type = self.data_frame.iloc[idx, 8]
        
        # Convert target based on binary mode
        if self.binary_mode:
            target = torch.tensor(target, dtype=torch.float).unsqueeze(0)
        else:
            target = torch.tensor(target, dtype=torch.long)
        
        # Apply transforms
        if isinstance(self.transform, list):
            image = self.transform[transform_type](image)
        else:
            image = self.transform(image)
            
        return image, target

def create_transforms(is_training=True):
    """
    Create data transformations for training or evaluation.
    
    Args:
        is_training (bool): Whether to create training or evaluation transforms
    
    Returns:
        transforms.Compose: Transformation pipeline
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor()
        ])