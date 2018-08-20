from utils.augmentation import BaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSet
from model.base_model import BaseModel
from utils.func import save_img, generate_pred_target_pkl
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import pickle
import os


if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    MEAN = [104.00698793, 116.66876762, 122.67891434]

    val = pickle.load(open(os.path.join(args.data_root, 'val.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'train', 'images')

    val_dataset = SaltSet(val, image_root, BaseTransform(args.size, MEAN, None))

    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=False)

    model = BaseModel(args)
    model.init_model()
    model.load_trained_model()

    pred, true = model.test_val(val_dataloader)

    pred_all = np.argmax(np.array(pred), 1)  #(N, 128, 128)
    target_all = np.array(true).astype(np.int) # (N, 128, 128)
    generate_pred_target_pkl(pred_all, target_all)

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
            tmp = np.transpose(pred[i], [1, 2, 0])
            tmp = tmp.reshape([tmp.shape[0], -1])
            save_img(tmp > 0.5, os.path.join(OUT_1, name + '_pred.jpg'))
            tmp = np.transpose(true[i], [1, 2, 0])
            tmp = tmp.reshape([tmp.shape[0], -1])
            save_img(tmp, os.path.join(OUT_2, name + '_true.jpg'))










