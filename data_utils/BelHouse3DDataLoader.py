import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter


class BelHouse3DSemSegDataset(Dataset):
    def __init__(self, root, split='train', num_points=2048, use_blocks=True, transform=None):

        self.root = root
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.use_blocks = use_blocks

        subdir = 'blocks' if use_blocks else 'rooms'
        search_path = os.path.join(root, split, subdir, '*.npy')
        self.files = glob.glob(search_path)

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {search_path}")

        self.class_weights = self.calculate_class_weights()

    def calculate_class_weights(self):
        label_counts = Counter()
        for file_path in self.files:
            data = np.load(file_path)
            labels = data[:, 3].astype(np.int64)
            label_counts.update(labels)

        total_count = sum(label_counts.values())
        num_classes = max(label_counts.keys()) + 1 

        freq = np.zeros(num_classes)
        for cls in range(num_classes):
            freq[cls] = label_counts.get(cls, 0) / total_count

        weights = 1 / (freq + 1e-6)

        weights = weights / weights.max()

        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path) 

        if self.use_blocks:
            if data.shape[0] >= self.num_points:
                indices = np.random.choice(data.shape[0], self.num_points, replace=False)
            else:
                indices = np.random.choice(data.shape[0], self.num_points, replace=True)
            data = data[indices]

        points = data[:, :3]
        labels = data[:, 3].astype(np.int64)

        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()
