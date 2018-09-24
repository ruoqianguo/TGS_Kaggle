from utils.augmentation import BaseTransform, HengBaseTransform
from options.base_options import SegOptions
from dataset.salt_set import SaltSetDeploy
from model.base_model import BaseModel
from utils.func import save_img
import torch.utils.data as data
from rlen import make_submission
from utils.tta import tta_collate
from torch.utils.data.dataloader import default_collate
import pickle
import os
import numpy as np

if __name__ == '__main__':
    options = SegOptions()
    args = options.parse()
    options.setup_option()

    ensemble_dict = 'ensemble_dict_eval'
    if not os.path.exists(ensemble_dict):
        os.mkdir(ensemble_dict)

    csv_name = args.exp_name

    MEAN = [104.00698793, 116.66876762, 122.67891434]

    test = pickle.load(open(os.path.join(args.data_root, 'test.pkl'), 'rb'))
    image_root = os.path.join(args.data_root, 'test', 'images')

    if args.aug == 'heng':
        base_aug = HengBaseTransform(MEAN)
    elif args.aug == 'default':
        base_aug = BaseTransform(args.size, MEAN, None)
    else:
        raise NotImplemented

    test_dataset = SaltSetDeploy(test, image_root, base_aug, use_depth=args.use_depth)

    preds_test = np.zeros((len(test_dataset), 101, 101))
    for ensemble_exp, ensemble_model, ensemble_snapshot, tta, ms in zip(args.ensemble_exp, args.ensemble_model,
                                                                        args.ensemble_snapshot, args.ensemble_tta,
                                                                        args.ensemble_ms):
        args.exp_name = ensemble_exp
        args.model_name = ensemble_model
        args.trained_model = ensemble_snapshot
        args.use_tta = tta
        args.ms = ms
        pkl_name = os.path.join(ensemble_dict,
                                ensemble_exp + '_' + ensemble_snapshot + '_' + str(tta) + '_' + str(ms) + '.pkl')
        print(pkl_name)

        if os.path.exists(
                os.path.join(ensemble_dict,
                             ensemble_exp + '_' + ensemble_snapshot + '_' + str(tta) + '_' + str(ms) + '.pkl')):
            d = pickle.load(open(pkl_name, 'rb'))
            pred = d['pred']
        else:
            test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                              pin_memory=True,
                                              collate_fn=tta_collate if args.use_tta else default_collate,
                                              shuffle=False)
            model = BaseModel(args)
            model.init_model()
            model.load_trained_model()

            pred = model.eval(test_dataloader)  # after softmax array
            pickle.dump({'pred': pred}, open(pkl_name, 'wb'))

        preds_test += pred

    preds_test /= len(args.ensemble_exp)
    # generate csv
    print('generate csv ...')
    for t in [0.23, 0.24]:
        make_submission((preds_test > t).astype(np.uint8), test['names'],
                        path='{}_{:.2f}_submission.csv'.format(csv_name, t))
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
