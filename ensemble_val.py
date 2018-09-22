from utils.augmentation import BaseTransform, HengBaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSet
from model.base_model import BaseModel
from utils.func import save_img, generate_pred_target_pkl
import torch.utils.data as data
from tqdm import tqdm
from utils.tta import tta_collate
from torch.utils.data.dataloader import default_collate
import pickle
import os
import numpy as np
from utils.metrics import mIoU

if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    ensemble_dict = 'ensemble_dict'
    if not os.path.exists(ensemble_dict):
        os.mkdir(ensemble_dict)

    MEAN = [104.00698793, 116.66876762, 122.67891434]

    val = pickle.load(open(os.path.join(args.data_root, 'val.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'train', 'images')

    if args.aug == 'heng':
        base_aug = HengBaseTransform(MEAN)
    elif args.aug == 'default':
        base_aug = BaseTransform(args.size, MEAN, None)
    else:
        raise NotImplemented

    val_dataset = SaltSet(val, image_root, base_aug, use_depth=args.use_depth, original_mask=True)

    all_pred = np.zeros((len(val_dataset), 101, 101))
    all_true = None
    for ensemble_exp, ensemble_model, ensemble_snapshot, tta, ms in zip(args.ensemble_exp, args.ensemble_model,
                                                                        args.ensemble_snapshot, args.ensemble_tta,
                                                                        args.ensemble_ms):
        args.exp_name = ensemble_exp
        args.model_name = ensemble_model
        args.trained_model = ensemble_snapshot
        args.use_tta = tta
        args.ms = ms
        pkl_name = os.path.join(ensemble_dict, ensemble_exp + '_' + ensemble_snapshot + '_' + str(tta) + '_' + str(ms) + '.pkl')
        print(pkl_name)

        if os.path.exists(
                os.path.join(ensemble_dict, ensemble_exp + '_' + ensemble_snapshot + '_' + str(tta) + '_' + str(ms) + '.pkl')):
            d = pickle.load(open(pkl_name, 'rb'))
            pred = d['pred']
            true = d['true']
        else:
            val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=tta_collate if args.use_tta else default_collate,
                                             shuffle=False)
            model = BaseModel(args)
            model.init_model()
            model.load_trained_model()

            pred, true = model.test_val(val_dataloader)  # after softmax
            pred = np.array(pred)
            true = np.array(true)
            pickle.dump({'pred': pred, 'true': true}, open(pkl_name, 'wb'))

        all_pred += pred
        if args.vis:
            print('vis mask ...')
            OUT_1 = os.path.join('Visualize', args.exp_name + '_pred')
            OUT_2 = os.path.join('Visualize', args.exp_name + '_mask')
            if not os.path.exists(OUT_1):
                os.makedirs(OUT_1)
            if not os.path.exists(OUT_2):
                os.makedirs(OUT_2)

            for i, d in tqdm(enumerate(val['names'])):
                name = d
                tmp = pred[i]
                save_img(tmp > 0.5, os.path.join(OUT_1, name + '_pred.jpg'))
                tmp = true[i]
                save_img(tmp, os.path.join(OUT_2, name + '_true.jpg'))

    all_true = true
    all_pred /= len(args.ensemble_exp)
    print('ensemble results')
    for t in np.arange(0.25, 0.55, 0.01):
        pred_all = all_pred > t
        true_all = all_true.astype(np.int)
        # new_iou = intersection_over_union(true_all, pred_all)
        # new_iou_t = intersection_over_union_thresholds(true_all, pred_all)
        mean_iou, iou_t = mIoU(true_all, pred_all)
        print('threshold : {:.4f}'.format(t))
        print('mean IoU : {:.4f}, IoU threshold : {:.4f}'.format(mean_iou, iou_t))
