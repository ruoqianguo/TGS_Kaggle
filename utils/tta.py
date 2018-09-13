import cv2
import torch
import numpy as np

tta_config = ['origin', 'mirror']
MEAN = [104.00698793, 116.66876762, 122.67891434]


def decode_normal(image):
    """ Decode normalization operation
    :param image: numpy.array, [H, W, 3]
    :return: numpy.array, [H, W, 3]
    """
    return image + MEAN


def normalize(image):
    """ normalization operation
    :param image: numpy.array, [H, W, 3]
    :return: numpy.array, [H, W, 3]
    """
    return image - MEAN


ttas = {
    'origin': lambda img: img,
    'rot90': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    'rot180': lambda img: cv2.rotate(img, cv2.ROTATE_180),
    'rot270': lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    'flip': lambda img: cv2.flip(img, 0),
    'mirror': lambda img: cv2.flip(img, 1)
}


detta = {
    'origin': lambda img: img,
    'rot90': lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    'rot180': lambda img: cv2.rotate(img, cv2.ROTATE_180),
    'rot270': lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
    'flip': lambda img: cv2.flip(img, 0),
    'mirror': lambda img: cv2.flip(img, 1)
}


def tta_collate(batch):
    if isinstance(batch[0], tuple):
        images = []  # [tta x bs, 3, H, W]
        masks = []
        for i, tta in enumerate(tta_config):
            tta_func = ttas[tta]
            for t in batch:
                image = t[0].numpy()
                image = decode_normal(image.transpose(1, 2, 0))
                image = tta_func(image)
                image = normalize(image).transpose(2, 0, 1)
                images.append(image.astype(np.float32))
                if i == 0:
                    masks.append(t[1])
        images = np.stack(images)
        masks = np.stack(masks)
        return torch.from_numpy(images), torch.from_numpy(masks)
    else:
        images = []  # [tta x bs, 3, H, W]
        for tta in tta_config:
            tta_func = ttas[tta]
            for image in batch:
                image = decode_normal(image.numpy().transpose(1, 2, 0))
                image = tta_func(image)
                image = normalize(image).transpose(2, 0, 1)
                images.append(image.astype(np.float32))  # (3, H, W)
        images = np.stack(images)

        return torch.from_numpy(images)  # [tta x bs, 3, H, W]


def detta_score(score):
    """
    :param score: Variable, [num_tta, bs, nclass, H, W]
    :return: Variable, [num_tta, bs, nclass, H, W]
    """
    num_tta, bs, n_class, H, W = score.size()
    score = score.detach()

    for t in range(num_tta):
        dt = detta[tta_config[t]]
        for b in range(bs):
            reverse = dt(score[t, b].cpu().data.numpy().transpose(1, 2, 0))  # [H, W, nclass]
            reverse = np.ascontiguousarray(reverse.transpose(2, 0, 1))  # [nclass, H, W]
            reverse = torch.from_numpy(reverse)
            score[t, b].data.copy_(reverse)
    return score