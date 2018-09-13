import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.func import one_hot_embedding
from torch.nn import BCELoss, CrossEntropyLoss
from model.lovasz_loss import lovasz_softmax

class DiceLoss(nn.Module):

    def __init__(self, num_classes, smooth=1):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, input, target):
        """
        :param input: torch.FloatTensor, [bs, num_classes, H, W]
        :param target: torch.LongTensor, [bs, H, W]
        :return: dice loss
        """
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        input = input.float()  # [bs x H x W, num_classes]
        target = target.view(-1)
        target = one_hot_embedding(target, self.num_classes)  # [bs x H x W, num_classes]
        target = target.float()

        intersect = input * target
        score = (2. * intersect.sum() + self.smooth) / (input.pow(2).sum() + target.pow(2).sum() + self.smooth)
        return 1 - score

class MixLoss(nn.Module):

    def __init__(self, num_classes, smooth=1, weights=[1.0, 1.0]):
        super(MixLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.diceloss = DiceLoss(num_classes, smooth)
        self.celoss = CrossEntropyLoss()
        self.weights = weights

    def forward(self, input, target):
        """
        :param input: torch.FloatTensor, [bs, num_classes, H, W]
        :param target: torch.LongTensor, [bs, H, W]
        :return: dice loss
        """
        diceloss = self.diceloss(input, target)
        celoss = self.celoss(input, target)
        return celoss * self.weights[0] + diceloss * self.weights[1]


class LovaszSoftmax(nn.Module):
    def __init__(self):
        super(LovaszSoftmax, self).__init__()
        self.lovasz_softmax = lovasz_softmax

    def forward(self, pred, label):
        """
        :param pred:  b, c, h, w
        :param label:  b, h, w
        :return:
        """
        pred = F.softmax(pred, dim=1)
        return self.lovasz_softmax(pred, label)


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha, gamma):
        super(FocalLoss, self).__init__()

        self.num_classes = num_classes

        if alpha is not None:
            self.alpha = torch.ones(num_classes) * alpha
            self.alpha[0] = 1 - alpha
            self.alpha = Variable(self.alpha.cuda(), requires_grad=False)
        else:
            self.alpha = Variable(torch.ones(num_classes).cuda(), requires_grad=False)

        self.gamma = gamma

    def forward(self, pred, label):
        '''Focal loss.

        Args:
          pred: (tensor) sized [b, c, h, w]
          label: (tensor) sized [b, h, w].

        Return:
          (tensor) focal loss.
        '''

        x = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
        y = label.view(-1)

        p = F.softmax(x)
        p = p.gather(1, y.view(-1, 1)).view(-1)
        log_p = (p + 1e-6).log()

        alpha = self.alpha[y.long().view(-1)]
        # w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = alpha * (1 - p.detach()).pow(self.gamma)
        loss = -(w * log_p)
        # print('w', w)
        # print('p', p)
        # print('################################')
        return loss.mean()
