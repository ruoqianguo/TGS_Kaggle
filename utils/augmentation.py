import numpy as np
from PIL import Image
import math
import cv2


class RandomScale(object):
    def __init__(self, low=0.5, high=1.5):
        self.low = low
        self.high = high

    def __call__(self, image, mask=None):
        f_scale = np.random.randint(int(self.low * 10), int(self.high * 10) + 1) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, mask


class Crop(object):
    def __init__(self, crop_h=513, crop_w=513, ignore_label=255, random=True):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.ignore_label = ignore_label
        self.random = random

    def __call__(self, image, mask=None):
        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            if mask is not None:
                mask_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, mask_pad = image, mask
        img_h, img_w, _ = img_pad.shape
        if self.random:
            h_off = np.random.randint(0, img_h - self.crop_h + 1)
            w_off = np.random.randint(0, img_w - self.crop_w + 1)
        else:
            h_off = 0
            w_off = 0
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        if mask is not None:
            mask = np.asarray(mask_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        return image, mask


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
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)


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
            expand_mask = np.zeros((int(height * ratio), int(width * ratio)), dtype=np.float32)
            expand_mask[:, :] = 0
            expand_mask[int(top):int(top + height),
            int(left):int(left + width)] = mask
            mask = expand_mask

        return image, mask


class BasePadding(object):
    def __init__(self, size=128, fill=0):
        self.fill = fill
        self.size = size

    def __call__(self, image, mask=None):
        h, w, c = image.shape

        expand_image = np.zeros((self.size, self.size, c), dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        left = int((self.size - w) / 2)
        top = int((self.size - h) / 2)
        expand_image[top: h + top, left: w + left, :] = image
        image = expand_image

        if mask is not None:
            expand_mask = np.zeros((self.size, self.size), dtype=mask.dtype)
            expand_mask[top:top + h, left:left + w] = mask
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
            for t in range(50):
                i, j, h, w = self.get_params(image, self.scale, self.ratio)
                cropped = image[i:i + h, j:j + w, :]
                if mask is not None:
                    crop_mask = mask[i:i + h, j:j + w]
                    if np.all(crop_mask == -255):
                        continue
                    mask = cv2.resize(crop_mask, self.size, interpolation=cv2.INTER_NEAREST)
                    break
                else:
                    break
            return cv2.resize(cropped, self.size), mask

        h, w, _ = image.shape
        if (h == self.size) and (w == self.size):
            return image, mask
        else:
            if mask is not None:
                mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            return cv2.resize(image, self.size), mask


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


class RandomGamma(object):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    # expects float image
    def __call__(self, image, mask=None):
        if np.random.randint(2):
            gamma = np.random.uniform(1-self.gamma, 1+self.gamma)
            image **= (1.0/gamma)
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


class HorizontalShear(object):
    def __init__(self, low=-0.07, high=0.07):
        self.low = low
        self.high = high

    def __call__(self, image, mask):
        if np.random.randint(2):
            return image, mask

        dx = np.random.uniform(self.low, self.high)
        borderMode = cv2.BORDER_REFLECT_101
        height, width = image.shape[:2]
        dx = int(dx * width)

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
        box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                                   borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        #         mask  = (mask>0.5).astype(np.float32)

        return image, mask


class RandomShiftScaleCropPad(object):
    def __init__(self, limit=0.2):
        self.limit = limit

    def _do_shift_scale_crop(self, image, mask, x0, y0, x1, y1):
        height, width = image.shape[:2]
        image = image[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]

        image = cv2.resize(image, dsize=(width, height))
        mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        #         mask  = (mask>0.5).astype(np.float32)
        return image, mask

    def __call__(self, image, mask):
        if np.random.randint(2):
            return image, mask

        H, W = image.shape[:2]

        dy = int(H * self.limit)
        y0 = np.random.randint(0, dy)
        y1 = H - np.random.randint(0, dy)

        dx = int(W * self.limit)
        x0 = np.random.randint(0, dx)
        x1 = W - np.random.randint(0, dx)
        image, mask = self._do_shift_scale_crop(image, mask, x0, y0, x1, y1)
        return image, mask


class ShiftScaleRotate(object):
    def __init__(self, scale=1, angle_max=15, dx=0, dy=0):
        self.scale = scale
        self.angle_max = angle_max
        self.dx = dx
        self.dy = dy

    def __call__(self, image, mask):
        if np.random.randint(2):
            return image, mask

        borderMode = cv2.BORDER_REFLECT_101
        # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

        height, width = image.shape[:2]
        sx = self.scale
        sy = self.scale
        angle = np.random.uniform(0, self.angle_max)
        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + self.dx, height / 2 + self.dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                                   borderMode=borderMode, borderValue=(
            0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
        #         mask  = (mask>0.5).astype(np.float32)
        return image, mask


class Elastic(object):
    def __init__(self, grid=10, distort=0.15):
        self.grid = grid
        self.distort = distort

    def __call__(self, image, mask):
        if np.random.randint(2):
            return image, mask

        borderMode = cv2.BORDER_REFLECT_101
        height, width = image.shape[:2]

        x_step = int(self.grid)
        xx = np.zeros(width, np.float32)
        prev = 0
        for x in range(0, width, x_step):
            start = x
            end = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * (1 + np.random.uniform(-self.distort, self.distort))

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        y_step = int(self.grid)
        yy = np.zeros(height, np.float32)
        prev = 0
        for y in range(0, height, y_step):
            start = y
            end = y + y_step
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + y_step * (1 + np.random.uniform(-self.distort, self.distort))

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        # grid
        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,
                          borderValue=(0, 0, 0,))

        mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=borderMode,
                         borderValue=(0, 0, 0,))
        #         mask  = (mask>0.5).astype(np.float32)
        return image, mask


class CenterPadding(object):
    def __init__(self, factor=32):
        self.factor = factor

    def _compute_center_pad(self, H, W, factor=32):
        if H % factor == 0:
            dy0, dy1 = 0, 0
        else:
            dy = factor - H % factor
            dy0 = dy // 2
            dy1 = dy - dy0

        if W % factor == 0:
            dx0, dx1 = 0, 0
        else:
            dx = factor - W % factor
            dx0 = dx // 2
            dx1 = dx - dx0
        return dy0, dy1, dx0, dx1

    def __call__(self, image, mask=None):
        H, W = image.shape[:2]
        dy0, dy1, dx0, dx1 = self._compute_center_pad(H, W, self.factor)

        image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
        # cv2.BORDER_CONSTANT, 0)
        if mask is not None:
            mask = cv2.copyMakeBorder(mask, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
        return image, mask


class HengAugmentation(object):
    def __init__(self, mean, crop_limit=0.2, shear_high=0.07, scale=1, angle=15,elastic_grid=10, elastic_distort=0.15, size=202, factor=32):
        self.horizontal_flip = RandomHorizontalFlip()
        self.deformation = [
            RandomShiftScaleCropPad(crop_limit),
            HorizontalShear(-shear_high, shear_high),
            ShiftScaleRotate(scale, angle),
            Elastic(elastic_grid, elastic_distort),
        ]
        self.bright_contrast = [
            RandomBrightness(25),
            RandomContrast(0.92, 1.08),
            RandomGamma(0.08),
        ]
        self.base_transofrm=Compose([
            Resize(size),
            CenterPadding(factor),
            NormalizeMean(mean),
        ])

    def __call__(self, image, mask):
        image, mask = self.horizontal_flip(image, mask)
        c = np.random.choice(4)
        image, mask = self.deformation[c](image, mask)
        c = np.random.choice(3)
        image, mask = self.bright_contrast[c](image, mask)
        image, mask = self.base_transofrm(image, mask)
        return image, mask


class HengBaseTransform(object):
    def __init__(self,  mean, size=202, factor=32):
        self.base_transofrm = Compose([
            Resize(size),
            CenterPadding(factor),
            NormalizeMean(mean),
        ])

    def __call__(self, image, mask=None):
        return self.base_transofrm(image, mask)


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


class BaseTransform2(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            BasePadding(size),
            NormalizeMean(mean)
        ])

    def __call__(self, image, mask=None):
        return self.augmentation(image, mask)


class VOCAugmentation(object):
    def __init__(self, mean, crop_h, crop_w, ignore_label, scale_low, scale_high):
        self.augmentation = Compose([
            RandomScale(scale_low, scale_high),
            RandomRotate(),
            RandomRotateAlpha(45),
            NormalizeMean(mean),
            Crop(crop_h, crop_w, ignore_label, random=True),
            RandomHorizontalFlip(),
        ])

    def __call__(self, image, mask):
        return self.augmentation(image, mask)


class VOCBaseTransform(object):
    def __init__(self, mean, crop_h, crop_w, ignore_label):
        self.augmentation = Compose([
            NormalizeMean(mean),
            Crop(crop_h, crop_w, ignore_label, random=False),
        ])

    def __call__(self, image, mask=None):
        return self.augmentation(image, mask)