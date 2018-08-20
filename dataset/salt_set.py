import torch.utils.data as data
import numpy as np
import torch


class SaltSet(data.Dataset):
    def __init__(self, data_label, image_root, augmentation):
        self.data_label = data_label
        self.image_root = image_root
        self.augmentation = augmentation

    def __getitem__(self, item):
        img = self.data_label['images'][item]
        mask = self.data_label['masks'][item]
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        img, mask = self.augmentation(img.astype(np.float32).copy(), mask.copy())
        # img = img[:, :, np.newaxis]
        mask = mask.astype(np.float32) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1), mask.astype(np.long)

    def __len__(self):
        return len(self.data_label['images'])

class SaltSetDeploy(data.Dataset):
    def __init__(self, data_label, image_root, augmentation):
        self.data_label = data_label
        self.image_root = image_root
        self.augmentation = augmentation

    def __getitem__(self, item):
        img = self.data_label['images'][item]
        img, mask = self.augmentation(img)
        img = img[:, :, np.newaxis]

        return torch.from_numpy(img).permute(2, 0, 1)

    def __len__(self):
        return len(self.data_label['images'])

