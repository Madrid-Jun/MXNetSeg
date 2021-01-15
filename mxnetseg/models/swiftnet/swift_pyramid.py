# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import FCNHead, AuxHead, ConvBlock

__all__ = ['get_swiftnet_pyramid', 'SwiftResNetPyramid']


class SwiftResNetPyramid(SegBaseResNet):
    """
    SwiftNetRN-18 with interleaved pyramid fusion.
    Reference: Orˇ, M., Kreˇ, I., & Bevandi, P. (2019). In Defense of Pre-trained ImageNet
        Architectures for Real-time Semantic Segmentation of Road-driving Images.
        In IEEE Conference on Computer Vision and Pattern Recognition.
    """

    def __init__(self, nclass, backbone='resnet18', aux=False, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(SwiftResNetPyramid, self).__init__(nclass, aux, backbone, height, width, base_size,
                                                 crop_size, pretrained_base, dilate=False,
                                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _SwiftNetHead(nclass, self._up_kwargs['height'], self._up_kwargs['width'],
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            if self.aux:
                self.auxlayer = AuxHead(nclass, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c2, c3, c4 = self.base_forward(x)
        x_half = F.contrib.BilinearResize2D(x, height=self._up_kwargs['height'] // 2,
                                            width=self._up_kwargs['width'] // 2)
        c1h, c2h, c3h, c4h = self.base_forward(x_half)
        outputs = []
        out = self.head(c4h, c3h, c2h, c1h, c4, c3, c2, c1)
        out = F.contrib.BilinearResize2D(out, **self._up_kwargs)
        outputs.append(out)
        # auxiliary loss on c3 of 1.0x scale
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        self.head.fusion_32x.up_kwargs['height'] = h // 32
        self.head.fusion_32x.up_kwargs['width'] = w // 32
        self.head.fusion_16x.up_kwargs['height'] = h // 16
        self.head.fusion_16x.up_kwargs['width'] = w // 16
        self.head.fusion_8x.up_kwargs['height'] = h // 8
        self.head.fusion_8x.up_kwargs['width'] = w // 8
        self.head.final.up_kwargs['height'] = h // 4
        self.head.final.up_kwargs['width'] = w // 4
        return self.forward(x)[0]


class _SwiftNetHead(nn.HybridBlock):
    """SwiftNet-Pyramid segmentation head"""

    def __init__(self, nclass, input_height, input_width, capacity=128, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_SwiftNetHead, self).__init__()
        with self.name_scope():
            self.conv1x1 = ConvBlock(capacity, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_32x = _LateralFusion(capacity, input_height // 32, input_width // 32,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_16x = _LateralFusion(capacity, input_height // 16, input_width // 16,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.fusion_8x = _LateralFusion(capacity, input_height // 8, input_width // 8,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.final = _LateralFusion(capacity, input_height // 4, input_width // 4, True,
                                        norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.seg_head = FCNHead(nclass, capacity, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c3h, c2h, c1h, c4, c3, c2, c1 = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
        c4h = self.conv1x1(x)
        out = self.fusion_32x(c4h, c3h, c4)
        out = self.fusion_16x(out, c2h, c3)
        out = self.fusion_8x(out, c1h, c2)
        out = self.final(out, c1)
        out = self.seg_head(out)
        return out


class _LateralFusion(nn.HybridBlock):
    """
    decoder up-sampling module.
    """

    def __init__(self, capacity, height, width, is_final=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(_LateralFusion, self).__init__()
        self.is_final = is_final
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1x1 = ConvBlock(capacity, 1, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')
            self.conv3x3 = ConvBlock(capacity, 3, 1, 1, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs, activation='relu')

    def hybrid_forward(self, F, x, *args, **kwargs):
        low = args[0] if self.is_final else F.concat(args[0], args[1], dim=1)
        low = self.conv1x1(low)
        high = F.contrib.BilinearResize2D(x, **self.up_kwargs)
        out = high + low
        out = self.conv3x3(out)
        return out


def get_swiftnet_pyramid(**kwargs):
    return SwiftResNetPyramid(**kwargs)
