import os
from pathlib import Path
from PIL import Image
# from skimage import io
import numpy as np
from tqdm import tqdm
import torch as t


def process(file_path, load_path=None, has_mask=True):
    print('start process ...')
    if load_path is not None:
        if os.path.exists(os.path.join(file_path, load_path)):
            datas = np.load(os.path.join(file_path, load_path))
            print('end process')
            return datas
        else:
            print(os.path.join(file_path, load_path) + ' is not exists')
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
        imgs = []
        for image in (file / 'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs) == 1
        if img.shape[2] > 3:
            assert (img[:, :, 3] != 255).sum() == 0
        img = img[:, :, :3]

        if has_mask:
            mask_files = list((file / 'masks').iterdir())
            masks = None
            for ii, mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask != 0)] == 255).all()
                if masks is None:
                    H, W = mask.shape
                    masks = np.zeros((len(mask_files), H, W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask != 0] == 255).all()
            mask = tmp_mask.astype(np.uint8)
            item['mask'] = mask
        item['name'] = str(file).split('/')[-1]
        item['img'] = img
        datas.append(item)
    np.save(os.path.join(file_path, load_path), datas)
    print('end process')
    return np.array(datas)

# You can skip this if you have alreadly done it.
# test = process('/home/grq/data/Bowl/stage1_test/',False)
# print(test)
# t.save(test, TEST_PATH)
# train_data = process('../input/stage1_train/')

# t.save(train_data, TRAIN_PATH)
