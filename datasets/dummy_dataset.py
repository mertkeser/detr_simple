from torch.utils.data import Dataset
import torch

class DummyDataset(Dataset):
    def __init__(self, length, num_classes=10):
        self.length = length
        self.images = torch.rand(length, 224, 224)
        self.labels = torch.randint(0, num_classes, (length,))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.length