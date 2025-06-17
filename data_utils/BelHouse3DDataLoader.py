import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class BelHouse3DSemSegDataset(Dataset):
    def __init__(self, root, split='train', num_points=2048, use_blocks=True, transform=None):
        """
        root: Path to IID-nonoccluded or OOD-occluded directory
        split: One of ['train', 'val', 'test']
        """
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
