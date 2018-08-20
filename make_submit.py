from utils.augmentation import BaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSet
from model.base_model import BaseModel
from utils.func import save_img, RLenc, upsample, downsample
import torch.utils.data as data
import numpy as np
import pickle
import os
import pandas as pd


if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    MEAN = [0.485, ]  # pytorch mean
    STD = [0.229, ]
    img_size_ori = 101
    img_size_target = args.size

    val = pickle.load(open(os.path.join(args.data_root, 'val.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'train', 'images')

    val_dataset = SaltSet(val, image_root, BaseTransform(args.size, MEAN, STD))

    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=False)

    model = BaseModel(args)
    model.init_model()
    model.load_trained_model()

    preds_test = model.eval(val_dataloader)

    pred_dict = {idx: RLenc(np.round(downsample(preds_test[i].squeeze(), img_size_ori, img_size_target) > 0.5)) for i, idx in
                 enumerate(val['names'])}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission.csv')











