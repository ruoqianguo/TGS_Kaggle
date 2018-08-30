import numpy as np
from PIL import Image
import math
import cv2


class AugmentColor(object):
    def __init__(self):
        self.U = np.array([[-0.56543481, 0.71983482, 0.40240142],
                           [-0.5989477, -0.02304967, -0.80036049],
                           [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32)
        self.EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
        self.sigma = 0.1
        self.color_vec = None

    def __call__(self, img, mask=None):
        color_vec = self.color_vec
        if self.color_vec is None:
            if not self.sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, self.sigma, 3)

        alpha = color_vec.astype(np.float32) * self.EV
        noise = np.dot(self.U, alpha.T) * 255
        return np.clip(img + noise[np.newaxis, np.newaxis, :], 0, 255), mask


class RandomHorizontalFlip(object):
    def __call__(self, img, mask=None):
        if np.random.randint(2):
            img = img[:, ::-1]
            if mask is not None:
                mask = mask[:, ::-1]
        return img, mask

class RandomVerticalFlip(object):
    def __call__(self, img, mask=None):
        if np.random.randint(2):
            img = img[::-1, :]
            if mask is not None:
                mask = mask[::-1, :]
        return img, mask


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, mask=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0, 255)
        return image, mask


class Padding(object):
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, mask=None):
        if np.random.randint(2):
            return image, mask

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 2.0)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image
        if mask is not None:
            expand_mask = np.zeros((int(height * ratio), int(width * ratio)), dtype=mask.dtype)
            expand_mask[:, :] = 0
            expand_mask[int(top):int(top + height),
            int(left):int(left + width)] = mask
            mask = expand_mask

        return image, mask


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, mask=None):
        if np.random.randint(2):
            h, w, _ = image.shape
            if (h == self.size) and (w == self.size):
                return image, mask
            else:
                if mask is not None:
                    mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
                return cv2.resize(image, self.size), mask

        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped = image[i:i + h, j:j + w, :]
        if mask is not None:
            crop_mask = mask[i:i+h, j:j+w]
            mask = cv2.resize(crop_mask, self.size, interpolation=cv2.INTER_NEAREST)
        return cv2.resize(cropped, self.size), mask


class RandomContrast(object):
    def __init__(self, lower=0.7, upper=1.3):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, mask=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
            image = np.clip(image, 0, 255)
        return image, mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class Rotate(object):
    def __call__(self, img, mask=None):
        angle = np.random.choice([0, -90, 90], 1)[0]
        delta = np.random.randint(-20, 20)
        angle += delta
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (cols, rows), borderValue=[0, 0, 0])
        return img, mask


class RandomRotate(object):

    def __init__(self):

        self.origin = lambda img: img
        self.rot90 = lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        self.rot180 = lambda img: cv2.rotate(img, cv2.ROTATE_180)
        self.rot270 = lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.rot_code = {
            0: self.origin,
            1: self.rot90,
            2: self.rot180,
            3: self.rot270
        }

    def __call__(self, image, labels):

        if len(image.shape) == 3:

            code = np.random.randint(4)
            image = np.ascontiguousarray(self.rot_code[code](image))
            labels = np.ascontiguousarray(self.rot_code[code](labels))
            return image, labels

        elif len(image.shape) == 4:

            for i in range(image.shape[0]):
                code = np.random.randint(2)
                image[i] = np.ascontiguousarray(self.rot_code[code](image[i]))
                labels[i] = np.ascontiguousarray(self.rot_code[code](labels[i]))

            return image, labels


class RandomRotateAlpha(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):

        if np.random.randint(2):
            return img, mask

        rotate_degree = np.random.random() * 2 * self.degree - self.degree
        img = Image.fromarray(img.astype(np.uint8))
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        return np.array(img), np.array(mask)


class AdaptiveHist(object):
    def __call__(self, im_cv, mask):
        im_cv = np.round(im_cv).astype(np.uint8)
        im = im_cv
        equalized_img = None
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(im_cv.shape) == 3:
            # im_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
            for i in range(3):
                im[:, :, i] = clahe.apply(im_cv[:, :, i])
            equalized_img = im
        # create a CLAHE object (Arguments are optional).
        else:
            equalized_img = clahe.apply(im_cv)
        # cv2.imwrite('tmp.jpg', equalized_img)
        return equalized_img.astype(np.float32), mask

class Normalize(object):
    def __init__(self, mean, std=None):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, mask=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, mask

class NormalizeMean(object):
    def __init__(self, mean):
        self.mean = np.array(mean)

    def __call__(self, image, mask=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image, mask

class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, mask=None):
        image = cv2.resize(image, (self.size,
                                   self.size))
        if mask is not None:
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, mask


class Augmentation(object):
    def __init__(self, size,  mean, std, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            RandomBrightness(),
            RandomContrast(),
            RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomRotate(),
            # RandomRotateAlpha(45),
            Padding(),
            RandomResizedCrop(self.size, self.scale, self.ratio),
            # Rotate(),
            NormalizeMean(mean)
        ])

    def __call__(self, image, mask=None):
        return self.augmentation(image, mask)

class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            NormalizeMean(mean)
        ])

    def __call__(self, image, mask=None):
        return self.augmentation(image, mask)