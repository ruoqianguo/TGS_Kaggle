from utils.summary import *
import time
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from model.unet import UNet
from model.unet_models import UNetResNet34, UNetResNet50, UNetResNet101, UNetResNet152, UNet11, UNetVGG16
from model.deeplab_v2 import deeplab_v2, deeplab50_v2, ms_deeplab_v2
from model.deeplab_v3 import deeplab_v3, ms_deeplab_v3
from model.loss import DiceLoss, MixLoss, LovaszSoftmax, FocalLoss
from utils.metrics import accuracy, mIoU, intersection_over_union_thresholds, intersection_over_union
from skimage.transform import resize
from utils.tta import tta_config, detta_score

def xavier(param):
    init.xavier_uniform(param)


def he(param):
    init.kaiming_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []
    try:
        b.append(model.conv1)
        b.append(model.bn1)
        b.append(model.layer1)
        b.append(model.layer2)
        b.append(model.layer3)
        b.append(model.layer4)   # test
    except AttributeError:
        b.append(model.Scale.conv1)
        b.append(model.Scale.bn1)
        b.append(model.Scale.layer1)
        b.append(model.Scale.layer2)
        b.append(model.Scale.layer3)
        b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    try:
        b.append(model.layer5.parameters())
        # b.append(model.layer4.parameters())
    except AttributeError:
        b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(base_lr, optimizer, i_iter, total_iters, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(base_lr, i_iter, total_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


class BaseModel:
    losses = {'train': [], 'val': []}
    acces = {'train': [], 'val': []}
    scores = {'train': [], 'val': []}
    pred = {'train': [], 'val': []}
    true = {'train': [], 'val': []}

    def __init__(self, args):
        self.args = args
        self.net = None
        print(args.model_name)
        if args.model_name == 'UNet':
            self.net = UNet(args.in_channels, args.num_classes)
            self.net.apply(weights_init)
        elif args.model_name == 'UNetResNet34':
            self.net = UNetResNet34(args.num_classes, dropout_2d=0.2)
        elif args.model_name == 'UNetResNet152':
            self.net = UNetResNet152(args.num_classes, dropout_2d=0.2)
        elif args.model_name == 'UNet11':
            self.net = UNet11(args.num_classes, pretrained=True)
        elif args.model_name == 'UNetVGG16':
            self.net = UNetVGG16(args.num_classes, pretrained=True, dropout_2d=0.0, is_deconv=True)
        elif args.model_name == 'deeplab50_v2':
            if args.ms:
                raise NotImplemented
            else:
                self.net = deeplab50_v2(args.num_classes, pretrained=args.pretrained)
        elif args.model_name == 'deeplab_v2':
            if args.ms:
                self.net = ms_deeplab_v2(args.num_classes, pretrained=args.pretrained, scales=args.ms_scales)
            else:
                self.net = deeplab_v2(args.num_classes, pretrained=args.pretrained)
        elif args.model_name == 'deeplab_v3':
            if args.ms:
                self.net = ms_deeplab_v3(args.num_classes, out_stride=args.out_stride, pretrained=args.pretrained, scales=args.ms_scales)
            else:
                self.net = deeplab_v3(args.num_classes, out_stride=args.out_stride, pretrained=args.pretrained)

        self.interp = nn.Upsample(size=args.size, mode='bilinear')

        self.iterations = args.epochs
        self.lr_current = args.lr
        self.cuda = args.cuda
        self.phase = args.phase
        if args.loss == 'CELoss':
            self.criterion = nn.CrossEntropyLoss(size_average=True)
        elif args.loss == 'DiceLoss':
            self.criterion = DiceLoss(num_classes=args.num_classes)
        elif args.loss == 'MixLoss':
            self.criterion = MixLoss(args.num_classes, weights=args.loss_weights)
        elif args.loss == 'LovaszLoss':
            self.criterion = LovaszSoftmax()
        elif args.loss == 'FocalLoss':
            self.criterion = FocalLoss(args.num_classes, 0.25, 2)
        else:
            raise RuntimeError('must define loss')

        if 'deeplab' in args.model_name:
            self.optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(self.net), 'lr': args.lr},
                                   {'params': get_10x_lr_params(self.net), 'lr': 10 * args.lr}],
                                  lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=args.lr,
                                   momentum=args.momentum, weight_decay=args.weight_decay)
        self.iters = 0

    def init_model(self):
        if self.args.resume_model:
            saved_state_dict = torch.load(self.args.resume_model, map_location=lambda storage, loc: storage)
            if self.args.ms:
                new_params = self.net.Scale.state_dict().copy()
                for i in saved_state_dict:
                    # Scale.layer5.conv2d_list.3.weight
                    i_parts = i.split('.')
                    # print i_parts
                    if not (i_parts[0] == 'layer5'):
                        new_params[i] = saved_state_dict[i]
                self.net.Scale.load_state_dict(new_params)
            else:
                new_params = self.net.state_dict().copy()
                for i in saved_state_dict:
                    # Scale.layer5.conv2d_list.3.weight
                    i_parts = i.split('.')
                    # print i_parts
                    if not (i_parts[0] == 'layer5'):
                    # if (not (i_parts[0] == 'layer5')) or (not (i_parts[0] == 'layer4')):
                        new_params[i] = saved_state_dict[i]
                self.net.load_state_dict(new_params)

            print('Resuming training, image net loading {}...'.format(self.args.resume_model))
            # self.load_weights(self.net, self.args.resume_model)

        if self.args.cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if epoch in self.args.stepvalues:
            self.lr_current = self.lr_current * self.args.gamma
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_current

    def save_network(self, net, net_name, epoch, label=''):
        save_fname = '%s_%s_%s.pth' % (epoch, net_name, label)
        save_path = os.path.join(self.args.save_folder, self.args.exp_name, save_fname)
        torch.save(net.state_dict(), save_path)

    def load_weights(self, net, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            net.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def load_trained_model(self):
        path = os.path.join(self.args.save_folder, self.args.exp_name, self.args.trained_model)
        print('eval cls, image net loading {}...'.format(path))
        if self.args.ms:
            self.load_weights(self.net.Scale, path)
        else:
            self.load_weights(self.net, path)

    def eval(self, dataloader):
        assert self.phase == 'test', "Command arg phase should be 'test'. "
        from tqdm import tqdm
        self.net.eval()
        output = []

        for i, image in tqdm(enumerate(dataloader)):
            if self.cuda:
                image = Variable(image.cuda(), volatile=True)
            else:
                image = Variable(image, volatile=True)

            # cls forward
            out = self.net(image)
            if isinstance(out, list):
                out_max = out[-1]
                if out_max.size(2) != image.size(2):
                    out = self.interp(out_max)
            else:
                if out.size(2) != image.size(2):
                    out = self.interp(out)
            # out [bs * num_tta, c, h, w]
            if self.args.use_tta:
                num_tta = len(tta_config)
                # out = F.softmax(out, dim=1)
                out = detta_score(out.view(num_tta, -1, self.args.num_classes, out.size(2),
                                           out.size(3)))  # [num_tta, bs, nclass, H, W]
                out = out.mean(dim=0)  # [bs, nclass, H, W]
            out = F.softmax(out)
            output.extend([resize(pred[1].data.cpu().numpy(), (101, 101)) for pred in out])
        return np.array(output)

    def tta(self, dataloaders):
        results = np.zeros(shape=(len(dataloaders[0].dataset), self.args.num_classes))
        for dataloader in dataloaders:
            output = self.eval(dataloader)
            results += output
        return np.argmax(results, 1)

    def tta_output(self, dataloaders):
        results = np.zeros(shape=(len(dataloaders[0].dataset), self.args.num_classes))
        for dataloader in dataloaders:
            output = self.eval(dataloader)
            results += output
        return results

    def test_val(self, dataloader):
        assert self.phase == 'test', "Command arg phase should be 'test'. "
        from tqdm import tqdm
        self.net.eval()
        predict = []
        true = []
        t1 = time.time()

        for i, (image, mask) in tqdm(enumerate(dataloader)):
            if self.cuda:
                image = Variable(image.cuda(), volatile=True)
                label_image = Variable(mask.cuda(), volatile=True)
            else:
                image = Variable(image, volatile=True)
                label_image = Variable(mask, volatile=True)

            # cls forward
            out = self.net(image)
            if isinstance(out, list):
                out_max = out[-1]
                if out_max.size(2) != label_image.size(2):
                    out = self.interp(out_max)
            else:
                if out.size(2) != image.size(2):
                    out = self.interp(out)
            # out [bs * num_tta, c, h, w]
            if self.args.use_tta:
                num_tta = len(tta_config)
                # out = F.softmax(out, dim=1)
                out = detta_score(out.view(num_tta, -1, self.args.num_classes, out.size(2),
                                           out.size(3)))  # [num_tta, bs, nclass, H, W]
                out = out.mean(dim=0)  # [bs, nclass, H, W]
            out = F.softmax(out)
            predict.extend([resize(pred[1].data.cpu().numpy(), (101, 101)) for pred in out])
            # predict.extend([pred[1, :101, :101].data.cpu().numpy() for pred in out])
            # pred.extend(out.data.cpu().numpy())
            true.extend(label_image.data.cpu().numpy())
        # pred_all = np.argmax(np.array(pred), 1)
        pred_all = np.array(predict) > 0.5
        true_all = np.array(true).astype(np.int)
        # new_iou = intersection_over_union(true_all, pred_all)
        # new_iou_t = intersection_over_union_thresholds(true_all, pred_all)
        mean_iou, iou_t = mIoU(true_all, pred_all)

        print('mean IoU : {:.4f}, IoU threshold : {:.4f}'.format(mean_iou, iou_t))

        return predict, true

    def run_epoch(self, dataloader, writer, epoch, train=True, metrics=True):
        if train:
            self.net.train()
            flag = 'train'
        else:
            self.net.eval()
            flag = 'val'
        t2 = time.time()
        for image, mask in dataloader:
            if train:
                adjust_learning_rate(self.args.lr, self.optimizer, self.iters, self.iterations * len(dataloader), 0.9)
                self.iters += 1

            if self.cuda:
                image = Variable(image.cuda(), volatile=(not train))
                label_image = Variable(mask.cuda(), volatile=(not train))
            else:
                image = Variable(image, volatile=(not train))
                label_image = Variable(mask, volatile=(not train))
            # cls forward
            out = self.net(image)

            if isinstance(out, list):
                out_max = None
                loss = 0.0
                for i, out_scale in enumerate(out):
                    if out_scale.size(2) != label_image.size(2):
                        out_scale = self.interp(out_scale)
                    if i == (len(out) - 1):
                        out_max = out_scale
                    loss += self.criterion(out_scale, label_image)
                label_image_np = label_image.data.cpu().numpy()
                sig_out_np = out_max.data.cpu().numpy()
                acc = accuracy(label_image_np, np.argmax(sig_out_np, 1))

                self.pred[flag].extend(sig_out_np)
                self.true[flag].extend(label_image_np)

                self.losses[flag].append(loss.data[0])
                self.acces[flag].append(acc)


            else:
                if out.size(-1) != label_image.size(-1):
                    out = self.interp(out)

                loss = self.criterion(out, label_image)
                label_image_np = label_image.data.cpu().numpy()
                sig_out_np = out.data.cpu().numpy()
                acc = accuracy(label_image_np, np.argmax(sig_out_np, 1))

                self.pred[flag].extend(sig_out_np)
                self.true[flag].extend(label_image_np)

                self.losses[flag].append(loss.data[0])
                self.acces[flag].append(acc)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if metrics:
            n = len(self.losses[flag])
            loss = sum(self.losses[flag]) / n
            scalars = [loss, ]
            names = ['loss', ]
            write_scalars(writer, scalars, names, epoch, tag=flag + '_loss')

            all_acc = sum(self.acces[flag]) / n
            scalars = [all_acc, ]
            names = ['all_acc', ]
            write_scalars(writer, scalars, names, epoch, tag=flag + '_acc')

            # all_score = sum(self.scores[flag]) / n
            # scalars = [all_score, ]
            # names = ['all_score', ]
            # write_scalars(writer, scalars, names, epoch, tag=flag + '_score')

            pred_all = np.argmax(np.array(self.pred[flag]), 1)
            true_all = np.array(self.true[flag]).astype(np.int)
            mean_iou, iou_t = mIoU(true_all, pred_all)

            # new_iou = intersection_over_union(true_all, pred_all)
            # new_iou_t = intersection_over_union_thresholds(true_all, pred_all)

            scalars = [mean_iou, iou_t, ]
            names = ['mIoU', 'mIoU_threshold', ]
            write_scalars(writer, scalars, names, epoch, tag=flag + '_IoU')

            print(
                '{} loss: {:.4f} | acc: {:.4f} | mIoU: {:.4f} | mIoU_threshold: {:.4f} |  n_iter: {} |  learning_rate: {} | time: {:.2f}'.format(flag, loss,
                                    all_acc, mean_iou, iou_t, epoch, self.optimizer.param_groups[0]['lr'], time.time() - t2))

            self.losses[flag] = []
            self.pred[flag] = []
            self.true[flag] = []
            self.acces[flag] = []
            self.scores[flag] = []

    def train_val(self, dataloader_train, dataloader_val, writer):
        val_epoch = 0
        for epoch in range(self.iterations):
            self.run_epoch(dataloader_train, writer, epoch, train=True, metrics=True)
            if (epoch+1) % self.args.save_freq == 0:
                self.run_epoch(dataloader_val, writer, val_epoch, train=False, metrics=True)
                val_epoch += 1

                if self.args.ms:
                    self.save_network(self.net.Scale, self.args.model_name, epoch=val_epoch, )
                else:
                    self.save_network(self.net, self.args.model_name, epoch=val_epoch, )
                print('saving in val_iteration {}'.format(val_epoch))
