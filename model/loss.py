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