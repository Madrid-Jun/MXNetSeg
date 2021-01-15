# coding=utf-8

from mxnet.gluon import nn
from ..base import SegBaseXception
from mxnetseg.nn import FCNHead, ASPP, ConvBlock

__all__ = ['get_deeplabv3plus', 'DeepLabv3Plus']


class DeepLabv3Plus(SegBaseXception):
    """
    DeepLab v3+ model.
    Reference: Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018).
        Encoder-decoder with atrous separable convolution for semantic image segmentation.
        In European Conference on Computer Vision (pp. 833–851).
        https://doi.org/10.1007/978-3-030-01234-2_49
    """

    def __init__(self, nclass, backbone='xception65', aux=True, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DeepLabv3Plus, self).__init__(nclass, aux, backbone, height, width, base_size,
                                            crop_size, pretrained_base, dilate=True,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _DeepLabHead(nclass, 2048, norm_layer, norm_kwargs,
                                     height=self._up_kwargs['height'] // 4,
                                     width=self._up_kwargs['width'] // 4)
            if self.aux:
                self.auxlayer = FCNHead(nclass, 728, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c1, c4)
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
        self.head.up_kwargs['height'] = h // 4
        self.head.up_kwargs['width'] = w // 4
        self.head.aspp.pool.up_kwargs['height'] = h // 8
        self.head.aspp.pool.up_kwargs['width'] = w // 8
        return self.forward(x)[0]


class _DeepLabHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, height=60, width=60):
        super(_DeepLabHead, self).__init__()
        self.up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.aspp = ASPP(256, in_channels, norm_layer, norm_kwargs, height // 2,
                             width // 2, atrous_rates=(12, 24, 36))
            self.conv_c1 = ConvBlock(48, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3x3 = ConvBlock(256, 3, 1, 1, in_channels=304, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
            self.drop = nn.Dropout(0.5)
            self.head = FCNHead(nclass, 256, norm_layer, norm_kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        c1, c4 = x, args[0]
        c1 = self.conv_c1(c1)
        out = self.aspp(c4)
        out = F.contrib.BilinearResize2D(out, **self.up_kwargs)
        out = F.concat(c1, out, dim=1)
        out = self.conv3x3(out)
        out = self.drop(out)
        return self.head(out)


def get_deeplabv3plus(**kwargs):
    return DeepLabv3Plus(**kwargs)
