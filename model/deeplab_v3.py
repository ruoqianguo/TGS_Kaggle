import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
affine_par = True

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

def outS(i):
    i = int(i)
    i = (i+1)//2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)//2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(nn.Sequential(*[
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(256),
        ]))

        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Sequential(*[
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True),
                nn.BatchNorm2d(256)
            ]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(2048, 256, 1, bias=True)
        self.pool_bn = nn.BatchNorm2d(256)
        self.aspp_conv = nn.Conv2d(256*5, 256, 1, bias=True)
        self.aspp_bn = nn.BatchNorm2d(256)
        self.logits = nn.Conv2d(256, num_classes, 1, bias=False)

        self.apply(weights_init)

    def forward(self, x):
        aspp_list = []
        for i in range(len(self.conv2d_list)):
            out = self.conv2d_list[i](x)
            aspp_list.append(out)

        # global image level
        pooled = self.avg_pool(x)
        pool_out = self.pool_bn(self.pool_conv(pooled))
        in_size = x.size()[2]
        resized = F.upsample(pool_out, size=in_size, mode='bilinear')
        aspp_list.append(resized)

        aspp_out = self.aspp_bn(self.aspp_conv(torch.cat(aspp_list, dim=1))) # bs, 256*5, h/16, w/16
        out = self.logits(aspp_out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, output_stride):
        self.inplanes = 64

        # define stride and rate
        # when output_stride = 16 ,stride=[2, 2, 1, 1], rate= [1,1,1,2]
        # when output_stride = 8, stride=[2, 1, 1, 1], rate = [1,1,2,4]
        if output_stride % 4 != 0:
            raise ValueError('The output_stride needs to be a multiple of 4.')
        self.out_stride = output_stride / 4
        stride = [2, 2, 2, 1]
        rate = [1, 1, 1, 1]
        current_rate = 1
        current_stride = 1
        for i, (s, r) in enumerate(zip(stride, rate)):
            if current_stride == self.out_stride:
                stride[i] = 1
                rate[i] = current_rate
                current_rate *= s
            else:
                current_stride *= s
        aspp_rate = [int(6 * (16 / output_stride)), int(12 * (16 / output_stride)), int(18 * (16 / output_stride))]

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change  out_stride=4
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0], dilation=rate[0])  # diff v2 stride=2 out_stride=8
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1], dilation=rate[1])  # out_stride=16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2], dilation=rate[2])  # out_stride=16
        self.layer4 = self._make_multi_grid_layer(block, 512, layers[3], stride=stride[3], dilation=rate[3], multi_grid=[1, 2, 4])
        self.layer5 = self._make_pred_layer(Classifier_Module, aspp_rate, aspp_rate, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False

        # self.bn1.apply(set_bn_fix)
        # self.layer1.apply(set_bn_fix)
        # self.layer2.apply(set_bn_fix)
        # self.layer3.apply(set_bn_fix)
        # self.layer4.apply(set_bn_fix)
        # self.layer5.apply(set_bn_fix)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_multi_grid_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=[1, 2, 4]):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation*multi_grid[0], downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride, dilation=dilation*multi_grid[i]))

        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


class MS_Deeplab(nn.Module):
    def __init__(self, model, scales):
        super(MS_Deeplab,self).__init__()
        self.Scale = model   #changed to fix #4
        self.scales = scales

    def forward(self, x):
        output = self.Scale(x) # for original scale
        output_size = output.size()[2]
        input_size = x.size()[2]
        interp_ori = nn.Upsample(size=(output_size, output_size), mode='bilinear')
        out_scales = [output, ]

        for i, s in enumerate(self.scales):
            if s == 1:
                continue
            else:
                interp = nn.Upsample(size=(int(input_size * s) + 1, int(input_size * s) + 1), mode='bilinear')
                x_scale = interp(x)
                output_scale = interp_ori(self.Scale(x_scale))
                out_scales.append(output_scale)

        out_max = output
        for out in out_scales:
            out_max = torch.max(out_max, out)

        out_scales.append(out_max)
        return out_scales


def ms_deeplab_v3(num_classes=21, out_stride=16, scales=[0.75, 1.0, 1.25], pretrained=True):
    model = MS_Deeplab(ResNet(Bottleneck, [3, 4, 23, 3], num_classes, out_stride), scales)
    if pretrained:
        pretrained_path = 'data/pretrained_model/VOC12_scenes_95_1321.pth'
        saved_state_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
        new_params = model.Scale.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not (i_parts[1] == 'layer5'):
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.Scale.load_state_dict(new_params)

    return model


def deeplab_v3(num_classes=21, out_stride=16, pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes, out_stride)
    if pretrained:
        pretrained_path = 'data/pretrained_model/VOC12_scenes_95_1321.pth'
        saved_state_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)['model']
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not (i_parts[0] == 'layer5'):
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

    return model


if __name__ == '__main__':
    deeplab = deeplab_v3()
    print(deeplab)
