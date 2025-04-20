import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets=None, sequence_length=6):
        self.features = torch.FloatTensor(features) if not isinstance(features, torch.Tensor) else features
        self.targets = torch.FloatTensor(targets) if targets is not None and not isinstance(targets,
                                                                                            torch.Tensor) else targets
        self.sequence_length = sequence_length
        self.has_targets = targets is not None

    def __len__(self):
        return max(0, len(self.features) - self.sequence_length + 1)

    def __getitem__(self, idx):
        if idx + self.sequence_length > len(self.features):
            raise IndexError("Index exceeds dataset length.")

        x = self.features[idx:idx + self.sequence_length]

        if self.has_targets:
            y = self.targets[idx + self.sequence_length - 1]
            return x, y
        else:
            return x