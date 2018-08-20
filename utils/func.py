from PIL import Image
import numpy as np
from skimage.transform import resize
import pickle

def save_img(img, path):
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    img.save(path)

def upsample(img, img_size_ori, img_size_target):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img, img_size_ori, img_size_target):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def make_submission(names, preds, path):
    """
    :param name: list ['11', '22', '33']
    :param pred: list [np.random.rand(101, 101), np.random.rand(101, 101)] 0, 1, 0
    :param name: 'submission.csv'
    :return:
    """
    pred_dict = {idx: RLenc(preds[i]) for i, idx in enumerate(names)}
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(path)


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def generate_pred_target_pkl(pred, target):
    """
    :param pred: bs, h, w
    :param target: bs, h, w
    :return:
    """
    out = {
        'pred':[],
        'target':[]
    }
    N = target.shape[0]
    for i in range(N):
        out['pred'].append(pred[i])
        out['target'].append(target[i])
    with open('pred_target.pkl', 'wb') as f:
        pickle.dump(out, f)
