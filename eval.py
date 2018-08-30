from utils.augmentation import BaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSetDeploy
from model.base_model import BaseModel
from utils.func import save_img, RLenc, downsample
from rlen import make_submission
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import os


if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    MEAN = [104.00698793, 116.66876762, 122.67891434]
    img_size_ori = 101

    test = pickle.load(open(os.path.join(args.data_root, 'test.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'test', 'images')

    test_dataset = SaltSetDeploy(test, image_root, BaseTransform(args.size, MEAN, None), use_depth=args.use_depth)

    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=False)

    model = BaseModel(args)
    model.init_model()
    model.load_trained_model()

    preds_test = model.eval(test_dataloader)  # (bs,101, 101)

    # generate csv
    print('generate csv ...')
    make_submission((preds_test>0.5).astype(np.uint8), test['names'], path='{}_submission.csv'.format(args.exp_name))
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










