import numpy as np
from pycocotools import mask as cocomask

# --------------------------------
def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)

def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations

def iou(gt, pred):
    gt[gt > 0] = 1.
    pred[pred > 0] = 1.
    intersection = gt * pred
    union = gt + pred
    union[union > 0] = 1.
    intersection = np.sum(intersection)
    union = np.sum(union)
    if union == 0:
        union = 1e-09
    return intersection / union


def compute_ious(gt, predictions):
    gt_ = get_segmentations(gt)
    predictions_ = get_segmentations(predictions)

    if len(gt_) == 0 and len(predictions_) == 0:
        return np.ones((1, 1))
    elif len(gt_) != 0 and len(predictions_) == 0:
        return np.zeros((1, 1))
    else:
        iscrowd = [0 for _ in predictions_]
        ious = cocomask.iou(gt_, predictions_, iscrowd)
        if not np.array(ious).size:
            ious = np.zeros((1, 1))
        return ious


def compute_precision_at(ious, threshold):
    mx1 = np.max(ious, axis=0)
    mx2 = np.max(ious, axis=1)
    tp = np.sum(mx2 >= threshold)
    fp = np.sum(mx2 < threshold)
    fn = np.sum(mx1 < threshold)
    return float(tp) / (tp + fp + fn)


def compute_eval_metric(gt, predictions):
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ious = compute_ious(gt, predictions)
    precisions = [compute_precision_at(ious, th) for th in thresholds]
    return sum(precisions) / len(precisions)


def intersection_over_union(y_true, y_pred):
    """
    :param y_true: shape (bs, 128, 128)
    :param y_pred: shape (bs, 128, 128)
    :return:
    """
    ious = []
    for y_t, y_p in list(zip(y_true, y_pred)):
        iou = compute_ious(y_t, y_p)
        iou_mean = 1.0 * np.sum(iou) / len(iou)
        ious.append(iou_mean)
    return np.mean(ious)


def intersection_over_union_thresholds(y_true, y_pred):
    """
        :param y_true: shape (bs, 128, 128)
        :param y_pred: shape (bs, 128, 128)
        :return:
    """
    iouts = []
    for y_t, y_p in list(zip(y_true, y_pred)):
        iouts.append(compute_eval_metric(y_t, y_p))
    return np.mean(iouts)
# --------------------







min_object_size = 1

def accuracy(true, pred, ignore=-1):
    valid = (true != ignore)
    return np.mean(true[valid] == pred[valid])

def mIoU(true, pred, ignore=-1):
    """
    :param true: shape (bs, 128, 128), dtype:0,1
    :param pred: shape (bs, 128, 128),
    :return: mIoU
    """
    bs = true.shape[0]
    smooth = 1e-6
    true = true.reshape(bs, -1)
    pred = pred.reshape(bs, -1)
    valid = (true != ignore)
    true = true * valid
    pred = pred * valid
    intersections = np.sum(true * pred, 1)
    union = np.sum(true, 1) + np.sum(pred, 1) - intersections
    iou = (intersections + smooth) / (union + smooth)

    thresholds = np.arange(0.5, 1.0, 0.05)
    count = 0.0
    for t in thresholds:
        count += np.mean(iou > t)
    iou_t = count / len(thresholds)


    # # tp compute is diff from normal tp
    # true_label = np.sum(true, 1) >= 0
    # m_iou = []
    # for t in thresholds:
    #     tf_label = iou > t
    #     tp = np.sum(true_label * tf_label)
    #     fp = np.sum((1-tf_label) * true_label)
    #     fn = np.sum((1-tf_label) * (1-true_label))
    #     p = (tp + smooth)/(tp + fp + fn + smooth)
    #     m_iou.append(p)

    return np.mean(iou), iou_t






