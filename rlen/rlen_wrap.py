import numpy as np
import pandas as pd
from .rlen import RLenc as fast_rlenc
from tqdm import tqdm


def list2txt(lst):
    z = ''
    for rr in lst:
        z += '{} {} '.format(rr[0], rr[1])
    return z[:-1]

# Source https://www.kaggle.com/bguberfain/unet-with-depth
# RLen python version

def RLenc_python(img, order='F', format=True):

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
        return list2txt(runs)
    else:
        return runs

def RLenc_cython(img, order='F', format=True):

    img = img.reshape(-1, order=order).astype(np.int32)
    # Cython accelerate RLenc
    res = fast_rlenc(img)
    if format:
        return list2txt(res)
    else:
        return res

def make_submission(preds, names, fast=True, path='submission.csv'):
    """
    :param preds: (list of np.array), [pred1, pred2, ...] each sized [H, W]
    :param names: (list), [name1, name2, ...]
    :param fast: (bool), flag of using Cython accelerate
    :param name: (str), path of submission, default = 'submission.csv'
    """
    RLenc = RLenc_cython if fast else RLenc_python

    rlen = []
    for pred in tqdm(preds):
        rlen.append(RLenc(pred))

    rlen_dict = {
        'id': names,
        'rle_mask': rlen
    }
    print('Exporting to {}.'.format(path))
    csv = pd.DataFrame(rlen_dict)
    csv.to_csv(path, index=None)
    print('Done.')
