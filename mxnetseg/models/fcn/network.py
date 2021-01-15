# coding=utf-8

from mxnet.gluon import nn
from mxnetseg.nn import FCNHead
from ..base import SegBaseResNet, SegBaseMobileNet

__all__ = ['get_fcn', 'FCNResNet', 'FCNMobileNet']


class FCNResNet(SegBaseResNet):
    """Fully Convolutional Networks based on ResNet"""

    def __init__(self, nclass, backbone='resnet50', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, dilate=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(FCNResNet, self).__init__(nclass, aux, backbone, height, width,
                                        base_size, crop_size, pretrained_base, dilate,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = FCNHead(nclass=nclass, in_channels=self.base_channels[3],
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.auxlayer = FCNHead(nclass, in_channels=self.base_channels[2],
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class FCNMobileNet(SegBaseMobileNet):
    """Fully Convolutional Networks based on MobileNet"""

    def __init__(self, nclass, backbone='mobilenet_v2_1_0', aux=True, height=None,
                 width=None, base_size=520, crop_size=480, pretrained_base=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(FCNMobileNet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                           crop_size, pretrained_base, norm_layer=norm_layer,
                                           norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = FCNHead(nclass=nclass, in_channels=self.base_channels[3],
                                norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.auxlayer = FCNHead(nclass=nclass, in_channels=self.base_channels[2],
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


def get_fcn(**kwargs):
    backbone_name = kwargs['backbone']
    if any(['resnet' in backbone_name, 'resnest' in backbone_name]):
        return FCNResNet(**kwargs)
    elif 'mobilenet' in backbone_name:
        return FCNMobileNet(**kwargs)
    else:
        raise NotImplementedError(f"Unknown backbone network: {kwargs['backbone']}")
