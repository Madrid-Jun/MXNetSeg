# coding=utf-8

from mxnet import init
from mxnet.gluon import nn
from ..base import SegBaseResNet
from mxnetseg.nn import SelfAttention, ConvBlock

__all__ = ['DANet', 'get_danet']


class DANet(SegBaseResNet):
    """
    ResNet based DANet.
    Reference: J. Fu et al., “Dual Attention Network for Scene Segmentation,”
        in IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, **kwargs):
        super(DANet, self).__init__(nclass, aux, backbone, height, width, base_size,
                                    crop_size, pretrained_base, dilate=True,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        with self.name_scope():
            self.head = _DANetHead(nclass, self.base_channels[3], norm_layer, norm_kwargs)
            if self.aux:
                self.auxlayer = _AuxLayer(nclass)

    def hybrid_forward(self, F, x, *args, **kwargs):
        _, _, _, c4 = self.base_forward(x)
        outputs = []
        out, pam_out, cam_out = self.head(c4)
        out = F.contrib.BilinearResize2D(out, **self._up_kwargs)
        outputs.append(out)

        if self.aux:
            pam_auxout, cam_auxout = self.auxlayer(pam_out, cam_out)
            pam_auxout = F.contrib.BilinearResize2D(pam_auxout, **self._up_kwargs)
            cam_auxout = F.contrib.BilinearResize2D(cam_auxout, **self._up_kwargs)
            outputs.append(pam_auxout)
            outputs.append(cam_auxout)

        return tuple(outputs)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        return self.forward(x)[0]


class _DANetHead(nn.HybridBlock):
    def __init__(self, nclass, in_channels, norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_DANetHead, self).__init__()
        inter_channels = in_channels // 4
        with self.name_scope():
            self.compress_pam = ConvBlock(inter_channels, 3, 1, 1, in_channels=in_channels,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.pam = SelfAttention(inter_channels)
            self.proj_pam = ConvBlock(inter_channels, 3, 1, 1, in_channels=inter_channels,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

            self.compress_cam = ConvBlock(inter_channels, 3, 1, 1, in_channels=in_channels,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.cam = _CAModule(inter_channels)
            self.proj_cam = ConvBlock(inter_channels, 3, 1, 1, in_channels=inter_channels,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs)

            self.head = nn.HybridSequential()
            self.head.add(nn.Dropout(0.1))
            self.head.add(nn.Conv2D(nclass, 1, in_channels=inter_channels))

    def hybrid_forward(self, F, x, *args, **kwargs):
        feat_1 = self.compress_pam(x)
        feat_1 = self.pam(feat_1)
        feat_1 = self.proj_pam(feat_1)

        feat_2 = self.compress_cam(x)
        feat_2 = self.cam(feat_2)
        feat_2 = self.proj_cam(feat_2)

        out = feat_1 + feat_2
        out = self.head(out)

        return out, feat_1, feat_2


class _AuxLayer(nn.HybridBlock):
    """
    Auxiliary loss layer for PAM and CAM.
    """

    def __init__(self, nclass):
        super(_AuxLayer, self).__init__()
        with self.name_scope():
            self.pam_layer = nn.HybridSequential()
            self.pam_layer.add(nn.Dropout(0.1))
            self.pam_layer.add(nn.Conv2D(nclass, 1))

            self.cam_layer = nn.HybridSequential()
            self.cam_layer.add(nn.Dropout(0.1))
            self.cam_layer.add(nn.Conv2D(nclass, 1))

    def hybrid_forward(self, F, x, *args, **kwargs):
        pam_aux = self.pam_layer(x)
        cam_aux = self.cam_layer(args[0])
        return pam_aux, cam_aux


class _CAModule(nn.HybridBlock):
    """
    Channel attention module.
    """

    def __init__(self, in_channels):
        super(_CAModule, self).__init__()
        self.in_channels = in_channels
        with self.name_scope():
            self.gamma = self.params.get('gamma', shape=(1,), init=init.Zero())

    def hybrid_forward(self, F, x, *args, **kwargs):
        gamma = kwargs['gamma']
        query = F.reshape(x, shape=(0, 0, -1))  # NC(HW)
        key = F.reshape(x, shape=(0, 0, -1))  # NC(HW)
        energy = F.batch_dot(query, key, transpose_b=True)  # NCC
        energy_new = F.max(energy, -1, True).broadcast_like(energy) - energy
        attention = F.softmax(energy_new)
        value = F.reshape(x, shape=(0, 0, -1))  # NC(HW)
        out = F.batch_dot(attention, value)  # # NC(HW)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)
        out = F.broadcast_mul(gamma, out) + x

        return out


def get_danet(**kwargs):
    return DANet(**kwargs)
