from utils.augmentation import BaseTransform, HengBaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSetDeploy
from model.base_model import BaseModel
from utils.func import save_img, RLenc, downsample
from rlen import make_submission
import torch.utils.data as data
from utils.tta import tta_collate
from torch.utils.data.dataloader import default_collate
import numpy as np
import pandas as pd
import pickle
import os


if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    ensemble_dict = 'ensemble_dict_eval'
    if not os.path.exists(ensemble_dict):
        os.mkdir(ensemble_dict)

    MEAN = [104.00698793, 116.66876762, 122.67891434]
    img_size_ori = 101

    test = pickle.load(open(os.path.join(args.data_root, 'test.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'test', 'images')

    if args.aug == 'heng':
        base_aug = HengBaseTransform(MEAN)
    elif args.aug == 'default':
        base_aug = BaseTransform(args.size, MEAN, None)
    else:
        raise NotImplemented

    test_dataset = SaltSetDeploy(test, image_root, base_aug, use_depth=args.use_depth)

    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True, collate_fn=tta_collate if args.use_tta else default_collate,
                                     shuffle=False)

    pkl_name = os.path.join(ensemble_dict,
                            args.exp_name + '_' + args.trained_model + '_' + str(args.use_tta) + '_' + str(args.ms) + '.pkl')
    print(pkl_name)

    if os.path.exists(pkl_name):
        d = pickle.load(open(pkl_name, 'rb'))
        preds_test = d['pred']
    else:
        model = BaseModel(args)
        model.init_model()
        model.load_trained_model()

        preds_test = model.eval(test_dataloader)  # (bs,101, 101)
        pickle.dump({'pred': preds_test}, open(pkl_name, 'wb'))

    # generate csv
    print('generate csv ...')
    for t in [0.14,]:
        make_submission((preds_test > t).astype(np.uint8), test['names'], path='{}_{:.2f}_submission.csv'.format(args.exp_name, t))
    # pred_dict = {idx: RLenc(np.round(preds_test[i] > 0.5)) for
    #              i, idx in tqdm(enumerate(test['names']))}
    #
    # sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    # sub.index.names = ['id']
    # sub.columns = ['rle_mask']
    # sub.to_csv('submission.csv')
    print('generate {}_submission.csv'.format(args.exp_name))

    if args.vis:
        print('vis mask ...')
        OUT_1 = os.path.join('Visualize', args.exp_name + '_eval')
        if not os.path.exists(OUT_1):
            os.makedirs(OUT_1)

        for i, d in enumerate(test['names']):
            name = d
            tmp = preds_test[i]
            save_img(tmp > 0.5, os.path.join(OUT_1, name + '_pred.jpg'))










