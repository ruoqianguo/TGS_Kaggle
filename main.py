from utils.augmentation import Augmentation, BaseTransform
from options.base_options import SegOptions
from tensorboardX import SummaryWriter
from dataset.salt_set import SaltSet
from model.base_model import BaseModel
import torch.utils.data as data
import pickle
import os


if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    writer = SummaryWriter(comment=args.exp_name)

    # MEAN = [0.485, 0.456, 0.406]  # pytorch mean
    # STD = [0.229, 0.224, 0.225]

    MEAN = [104.00698793, 116.66876762, 122.67891434]

    train = pickle.load(open(os.path.join(args.data_root, 'train.pkl'), 'rb'))
    val = pickle.load(open(os.path.join(args.data_root, 'val.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'train', 'images')

    train_dataset = SaltSet(train, image_root, Augmentation(args.size, MEAN, None))
    val_dataset = SaltSet(val, image_root, BaseTransform(args.size, MEAN, None))

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       pin_memory=True, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                                     shuffle=False)

    model = BaseModel(args)
    model.init_model()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    iters_per_epoch_val = int(len(val_dataset) / args.batch_size)
    print('train_epoch_size:{}, val_epoch_size:{}'.format(iters_per_epoch, iters_per_epoch_val))

    model.train_val(train_dataloader, val_dataloader, writer)






