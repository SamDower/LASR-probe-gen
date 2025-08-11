import torch
from torch.utils.data import Dataset


class ActivationDataset(Dataset):
    """Dataset class for activation data and labels"""

    def __init__(self, activations: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            activations: Tensor of shape (N, S, D) where N=samples, S=sequence_length, D=dimension
            labels: Tensor of shape (N,) with binary labels (0 or 1)
        """
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]
