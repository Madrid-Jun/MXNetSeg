# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import FCNHead, PyramidPooling, ConvBlock, LateralFusion

__all__ = ['get_swiftnet', 'SwiftResNet']


class SwiftResNet(SegBaseResNet):
    """
    ResNet based SwiftNet-Single.
    Reference: Orˇ, M., Kreˇ, I., & Bevandi, P. (2019). In Defense of Pre-trained ImageNet Architectures
        for Real-time Semantic Segmentation of Road-driving Images.
        In IEEE Conference on Computer Vision and Pattern Recognition.
    """

    def __init__(self, nclass, backbone='resnet18', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(SwiftResNet, self).__init__(nclass, aux, backbone, height, width, base_size, crop_size,
                                          pretrained_base, dilate=False, norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _SwiftNetHead(nclass, self.base_channels[3], norm_layer=norm_layer,
                                      norm_kwargs=norm_kwargs, input_height=self._up_kwargs['height'],
                                      input_width=self._up_kwargs['width'])
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
        self.head.ppool.up_kwargs['height'] = h // 32
        self.head.ppool.up_kwargs['width'] = w // 32
        self.head.fusion_c3.up_kwargs['height'] = h // 16
        self.head.fusion_c3.up_kwargs['width'] = w // 16
        self.head.fusion_c2.up_kwargs['height'] = h // 8
        self.head.fusion_c2.up_kwargs['width'] = w // 8
        self.head.fusion_c1.up_kwargs['height'] = h // 4
        self.head.fusion_c1.up_kwargs['width'] = w // 4
        return self.forward(x)[0]


class _SwiftNetHead(nn.HybridBlock):
    """SwiftNet segmentation head"""

    def __init__(self, nclass, in_channels, input_height, input_width, capacity=256,
                 norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_SwiftNetHead, self).__init__()
        with self.name_scope():
            self.ppool = PyramidPooling(in_channels, input_height // 32, input_width // 32,
                                        norm_layer, norm_kwargs)
            self.conv_c4 = ConvBlock(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

            self.fusion_c3 = LateralFusion(capacity, input_height // 16, input_width // 16,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_c2 = LateralFusion(capacity, input_height // 8, input_width // 8,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_c1 = LateralFusion(capacity, input_height // 4, input_width // 4,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.seg_head = FCNHead(nclass, capacity, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c4, c3, c2, c1 = x, args[0], args[1], args[2]
        c4 = self.ppool(c4)
        c4 = self.conv_c4(c4)
        out = self.fusion_c3(c4, c3)
        out = self.fusion_c2(out, c2)
        out = self.fusion_c1(out, c1)
        out = self.seg_head(out)
        return out


def get_swiftnet(**kwargs):
    return SwiftResNet(**kwargs)
