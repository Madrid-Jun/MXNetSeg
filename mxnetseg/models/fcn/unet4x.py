# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import ConvBlock, FCNHead, LateralFusion

__all__ = ['get_unet', 'UNet']


class UNet(SegBaseResNet):
    def __init__(self, nclass, backbone='resnet50', aux=False, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(UNet, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                   pretrained_base, dilate=False, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _NextHead(nclass, self._up_kwargs['height'], self._up_kwargs['width'],
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.auxlayer = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4, c3, c2, c1)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        self.head.fusion_16x.up_kwargs['height'] = h // 16
        self.head.fusion_16x.up_kwargs['width'] = w // 16
        self.head.fusion_8x.up_kwargs['height'] = h // 8
        self.head.fusion_8x.up_kwargs['width'] = w // 8
        self.head.fusion_4x.up_kwargs['height'] = h // 4
        self.head.fusion_4x.up_kwargs['width'] = w // 4
        return self.forward(x)[0]


class _NextHead(nn.HybridBlock):
    def __init__(self, nclass, input_height, input_width, capacity=256, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_NextHead, self).__init__()
        with self.name_scope():
            self.conv_c4 = ConvBlock(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_16x = LateralFusion(capacity, input_height // 16, input_width // 16,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_8x = LateralFusion(capacity, input_height // 8, input_width // 8,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_4x = LateralFusion(capacity, input_height // 4, input_width // 4,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.seg_head = FCNHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c4, c3, c2, c1 = x, args[0], args[1], args[2]
        c4 = self.conv_c4(c4)
        out = self.fusion_16x(c4, c3)
        out = self.fusion_8x(out, c2)
        out = self.fusion_4x(out, c1)
        out = self.seg_head(out)
        return out


def get_unet(**kwargs):
    return UNet(**kwargs)
