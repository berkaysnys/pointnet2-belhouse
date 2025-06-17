import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class BelHouse3DSemSegDataset(Dataset):
    def __init__(self, root, split='train', num_points=2048, transform=None):
        """
        root: Path to IID-nonoccluded or OOD-occluded directory
        split: One of ['train', 'val', 'test']
        """
        self.root = root
        self.split = split
        self.num_points = num_points
        self.transform = transform

        # Recursively find .npy files in blocks/ and rooms/
        search_path = os.path.join(root, split, '**', '*.npy')
        self.files = glob.glob(search_path, recursive=True)

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {search_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)  # Expected shape: (N, 4)

        if data.shape[0] >= self.num_points:
            indices = np.random.choice(data.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(data.shape[0], self.num_points, replace=True)

        sampled = data[indices]

        points = sampled[:, :3]  # x, y, z
        labels = sampled[:, 3].astype(np.int64)

        if self.transform:
            points = self.transform(points)

        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()
