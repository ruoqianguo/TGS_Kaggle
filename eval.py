from utils.augmentation import BaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSetDeploy
from model.base_model import BaseModel
from utils.func import save_img, RLenc, downsample
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

    MEAN = [0.485, ]  # pytorch mean
    STD = [0.229, ]
    img_size_ori = 101
    img_size_target = args.size

    test = pickle.load(open(os.path.join(args.data_root, 'test.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'test', 'images')

    test_dataset = SaltSetDeploy(test, image_root, BaseTransform(args.size, MEAN, STD))

    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=False)

    model = BaseModel(args)
    model.init_model()
    model.load_trained_model()

    preds_test = model.eval(test_dataloader)  # (bs, 1, 128, 128)

    # generate csv
    print('generate csv ...')
    pred_dict = {idx: RLenc(np.round(downsample(preds_test[i].squeeze(), img_size_ori, img_size_target) > 0.5)) for
                 i, idx in tqdm(enumerate(test['names']))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')
    print('generate {}_submission.csv'.format(args.exp_name))

    if args.vis:
        print('vis mask ...')
        OUT_1 = os.path.join('Visualize', args.exp_name + '_eval')
        if not os.path.exists(OUT_1):
            os.makedirs(OUT_1)

        for i, d in enumerate(test['names']):
            name = d
            tmp = np.transpose(preds_test[i], [1, 2, 0])
            tmp = tmp.reshape([tmp.shape[0], -1])
            save_img(tmp > 0.5, os.path.join(OUT_1, name + '_pred.jpg'))










