import torch.utils.data as data
import numpy as np
import torch
# TGS depth mean 255
depth_mean = 134.6669
# TGS depth std 255
depth_std = 55.4674
# TGS depth max
depth_max = 959.0


def add_depth_channels(image_np):
    h, w, _ = image_np.shape
    for row, const in enumerate(np.linspace(0, 1, h)):
        image_np[row, :, 1] = const
    image_np[:, :, 2] = image_np[:, :, 0] * image_np[:, :, 1]
    return image_np


class SaltSet(data.Dataset):
    def __init__(self, data_label, image_root, augmentation, use_depth=False, original_mask=False):
        self.data_label = data_label
        self.image_root = image_root
        self.augmentation = augmentation
        self.use_depth = use_depth
        self.original_mask = original_mask

    def __getitem__(self, item):
        img = self.data_label['images'][item]
        mask = self.data_label['masks'][item]
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        # img = add_depth_channels(img)
        if self.original_mask:
            img, _ = self.augmentation(img.astype(np.float32).copy(), None)
        else:
            img, mask = self.augmentation(img.astype(np.float32).copy(), mask.copy())
        if self.use_depth:
            depth = self.data_label['depths'][item]
            img[:, :, 2] = (depth / depth_max) * 255.0 - depth_mean
        # img = img[:, :, np.newaxis]
        mask = mask.astype(np.float32) / 255.0

        return torch.from_numpy(img).permute(2, 0, 1), mask.astype(np.long)

    def __len__(self):
        return len(self.data_label['images'])

class SaltSetDeploy(data.Dataset):
    def __init__(self, data_label, image_root, augmentation, use_depth=False):
        self.data_label = data_label
        self.image_root = image_root
        self.augmentation = augmentation
        self.use_depth = use_depth

    def __getitem__(self, item):
        img = self.data_label['images'][item]
        img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        img, mask = self.augmentation(img.astype(np.float32).copy())
        if self.use_depth:
            depth = self.data_label['depths'][item]
            img[:, :, 2] = (depth / depth_max) * 255.0 - depth_mean

        return torch.from_numpy(img).permute(2, 0, 1)

    def __len__(self):
        return len(self.data_label['images'])
