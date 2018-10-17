from utils.augmentation import Augmentation, BaseTransform, HengAugmentation, HengBaseTransform
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

    save_name = '{}_{}.pkl'
    train = pickle.load(
        open(os.path.join(args.data_root, 'kfold{}/'.format(args.total_fold), save_name.format('train', args.fold_index)), 'rb'))
    val = pickle.load(
        open(os.path.join(args.data_root, 'kfold{}/'.format(args.total_fold), save_name.format('val', args.fold_index)), 'rb'))
    print(val['names'][0:5], 'val', 'using {}/{} fold'.format(args.fold_index, args.total_fold))
    image_root = os.path.join(args.data_root, 'train', 'images')

    if args.aug == 'heng':
        aug = HengAugmentation(MEAN)
        base_aug = HengBaseTransform(MEAN)
    elif args.aug == 'default':
        aug = Augmentation(args.size, MEAN, None, scale=(0.1, 1.0))
        base_aug = BaseTransform(args.size, MEAN, None)
    else:
        raise NotImplemented

    train_dataset = SaltSet(train, image_root, aug, args.use_depth)
    val_dataset = SaltSet(val, image_root, base_aug, args.use_depth)

    # train_dataset = SaltSet(train, image_root, VOCAugmentation(MEAN, args.size, args.size, 0, 0.5, 1.5), args.use_depth)
    # val_dataset = SaltSet(val, image_root, VOCBaseTransform(MEAN, args.size, args.size, 0), args.use_depth)

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                       pin_memory=True, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     pin_memory=True,
                                     shuffle=False)

    model = BaseModel(args)
    model.init_model()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    iters_per_epoch_val = int(len(val_dataset) / args.batch_size)
    print('train_epoch_size:{}, val_epoch_size:{}'.format(iters_per_epoch, iters_per_epoch_val))

    model.train_val(train_dataloader, val_dataloader, writer)
