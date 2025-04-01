from torch.utils.data import BatchSampler, SequentialSampler
import numpy as np


class FixedRatioBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, target_class=1, target_ratio=0.1, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_class = target_class
        self.target_ratio = target_ratio
        self.drop_last = drop_last

        # Get all labels
        if hasattr(dataset, 'targets'):
            self.labels = dataset.targets
        else:
            self.labels = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                self.labels.append(label)
            self.labels = np.array(self.labels)

        # Get indices for each class
        self.target_indices = np.where(self.labels == target_class)[0]
        self.other_indices = np.where(self.labels != target_class)[0]

    def __iter__(self):
        # Calculate number of target samples per batch
        target_count = int(self.batch_size * self.target_ratio)
        other_count = self.batch_size - target_count

        # Shuffle indices
        target_indices = self.target_indices.copy()
        other_indices = self.other_indices.copy()
        np.random.shuffle(target_indices)
        np.random.shuffle(other_indices)

        # Create batches with fixed ratio
        target_idx, other_idx = 0, 0
        batches = []

        while True:
            # Check if we have enough samples left
            if (target_idx + target_count > len(target_indices) or
                    other_idx + other_count > len(other_indices)):
                # Not enough samples for a full batch
                if not self.drop_last and target_idx < len(target_indices) and other_idx < len(other_indices):
                    # Create a smaller final batch
                    remaining_target = len(target_indices) - target_idx
                    remaining_other = len(other_indices) - other_idx

                    batch = list(target_indices[target_idx:]) + list(
                        other_indices[other_idx:other_idx + min(remaining_other,
                                                                int(remaining_target * (
                                                                            1 - self.target_ratio) / self.target_ratio))])
                    batches.append(batch)
                break

            # Create a batch
            batch = list(target_indices[target_idx:target_idx + target_count]) + \
                    list(other_indices[other_idx:other_idx + other_count])
            batches.append(batch)

            # Update indices
            target_idx += target_count
            other_idx += other_count

        return iter(batches)

    def __len__(self):
        # Calculate an approximation of batch count
        target_count = int(self.batch_size * self.target_ratio)
        batches_from_target = len(self.target_indices) // target_count
        return batches_from_target



import torch
from torch.utils.data import DataLoader

# Assuming you have your dataset ready
# dataset = YourDataset(...)

# Create the custom batch sampler
batch_size = 32
batch_sampler = FixedRatioBatchSampler(
    dataset=dataset,
    batch_size=batch_size,
    target_class=1,
    target_ratio=0.1,
    drop_last=False
)

# Create the DataLoader with the batch sampler
dataloader = DataLoader(
    dataset=dataset,
    batch_sampler=batch_sampler,
    num_workers=4,
    pin_memory=True  # Helpful for GPU training
)

# Now you can use the dataloader in your training loop
for batch in dataloader:
    inputs, labels = batch
    # Your training code...