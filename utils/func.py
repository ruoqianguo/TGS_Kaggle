from PIL import Image
import numpy as np
from skimage.transform import resize
import pickle
import torch
from torch.autograd import Variable
import cv2


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


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    :param labels: (Variable - torch.LongTensor) class labels, sized [N,].
    :param num_classes: (int) number of classes.
    :return (Variable - torch.FloatTensor) encoded labels, sized [N, #classes].
    '''
    label = labels.data.long()
    eye = label.new(num_classes, num_classes)
    eye.copy_(torch.eye(num_classes))
    target_onehot = eye[label.view(-1)]  # [N, #classes]
    return Variable(target_onehot)


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
        'pred': [],
        'target': []
    }
    N = target.shape[0]
    for i in range(N):
        out['pred'].append(pred[i])
        out['target'].append(target[i])
    with open('pred_target.pkl', 'wb') as f:
        pickle.dump(out, f)


def get_mask_type(mask):
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(outer, border, border, border, border, borderType=cv2.BORDER_CONSTANT, value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0  # empty
    if cover == ((mask * outer) > 0.5).sum():
        return 1  # border
    if np.all(mask == mask[0]):
        return 2  # vertical

    percentage = cover / (101 * 101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


def get_fold_data(fold_index, total_fold, train_val):
    from sklearn.model_selection import StratifiedKFold
    masks = [mask.astype(np.float32) / 255.0 for mask in train_val['masks']]
    coverage_class = list(map(get_mask_type, masks))
    train_all = []
    evaluate_all = []
    skf = StratifiedKFold(n_splits=total_fold, random_state=1234, shuffle=True)
    for train_index, evaluate_index in skf.split(train_val['names'], coverage_class):
        train_all.append(train_index)
        evaluate_all.append(evaluate_index)
        print(train_index.shape, evaluate_index.shape)  # the shape is slightly different in different cv, it's OK
    train = {}
    val = {}
    for key, values in train_val.items():
        train[key] = list(np.array(values)[train_all[fold_index]])
        val[key] = list(np.array(values)[evaluate_all[fold_index]])
    print('using fold index: {:}'.format(fold_index))
    return train, val
