# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import ConvBlock, SeEModule, FCNHead

__all__ = ['get_seenet', 'SeENet']


class SeENet(SegBaseResNet):
    """
    Pang, Yanwei, et al. "Towards bridging semantic gap to improve semantic segmentation."
    Proceedings of the IEEE International Conference on Computer Vision. 2019.
    Based on dilated ResNet50/101/152.
    """

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm,norm_kwargs=None, **kwargs):
        super(SeENet, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                     pretrained_base, dilate=True, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _SeEHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 height=self._up_kwargs['height'] // 4,
                                 width=self._up_kwargs['width'] // 4)
            if self.aux:
                self.auxlayer = FCNHead(nclass, 1024, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c2, c3, c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)

    def predict(self, x):
        height, width = x.shape[2:]
        self._up_kwargs['height'] = height
        self._up_kwargs['width'] = width
        self.head.up_kwargs['height'] = height // 4
        self.head.up_kwargs['width'] = width // 4
        return self.forward(x)[0]


class _SeEHead(nn.HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 height=120, width=120):
        super(_SeEHead, self).__init__()
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.seemc2 = SeEModule(128, atrous_rates=(1, 2, 4, 8), norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, full_sample=False)
            self.seemc3 = SeEModule(128, atrous_rates=(3, 6, 9, 12), norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, full_sample=False)
            self.seemc4 = SeEModule(128, atrous_rates=(7, 13, 19, 25), norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs, full_sample=True)
            self.bam = _BoundaryAttention(nclass, low_channels=256, high_channels=128,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = x, args[0], args[1], args[2]
        c2 = self.seemc2(c2)
        c3 = self.seemc3(F.concat(c2, c3, dim=1))
        c4 = self.seemc4(F.concat(c3, c4, dim=1))
        c4 = F.contrib.BilinearResize2D(c4, **self.up_kwargs)
        out = self.bam(c4, c1)
        return out


class _BoundaryAttention(nn.HybridBlock):
    def __init__(self, nclass, low_channels=256, high_channels=128,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, drop=.1):
        super(_BoundaryAttention, self).__init__()
        with self.name_scope():
            self.conv1x1 = ConvBlock(low_channels, 1, in_channels=high_channels, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='sigmoid')
            self.fconv1x1 = ConvBlock(high_channels, 1, in_channels=low_channels,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fconv3x3 = ConvBlock(high_channels, 3, 1, 1, in_channels=high_channels,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.cconv3x3 = ConvBlock(high_channels, 3, 1, 1, in_channels=high_channels,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.drop = nn.Dropout(drop) if drop else None
            self.cconv1x1 = nn.Conv2D(nclass, 1, in_channels=high_channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        high, low = x, args[0]
        score = self.conv1x1(high)
        low = low * (1 - score)
        low = self.fconv1x1(low)
        low = self.fconv3x3(low)
        out = high + low
        out = self.cconv3x3(out)
        if self.drop:
            out = self.drop(out)
        return self.cconv1x1(out)


def get_seenet(**kwargs):
    return SeENet(**kwargs)
