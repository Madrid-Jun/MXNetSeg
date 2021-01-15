# coding=utf-8

import math
from mxnet import base
from mxnet.gluon import nn

__all__ = ['NonLocalBlock', 'SpatialPyramidPool']


class NonLocalBlock(nn.HybridBlock):
    """
    Non-local block.
    Reference: Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local Neural Networks.
        In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7794–7803).
        IEEE. https://doi.org/10.1109/CVPR.2018.00813
    """

    def __init__(self, in_channels, subsample=False, norm_layer=nn.BatchNorm, norm_kwargs=None, mode='em_gaussian'):
        super(NonLocalBlock, self).__init__()
        with self.name_scope():
            assert mode in ['dot_product', 'em_gaussian']
            self.inter_channels = in_channels // 2
            self.mode = mode

            self.g = nn.Conv2D(channels=self.inter_channels, kernel_size=1, use_bias=False, in_channels=in_channels)
            self.theta = nn.Conv2D(channels=self.inter_channels, kernel_size=1, use_bias=False, in_channels=in_channels)
            self.phi = nn.Conv2D(channels=self.inter_channels, kernel_size=1, use_bias=False, in_channels=in_channels)
            self.subsample = subsample
            if self.subsample:
                self.pool_layer = nn.MaxPool2D(pool_size=(2, 2))

            self.z = nn.Conv2D(channels=in_channels, kernel_size=1, use_bias=False, in_channels=self.inter_channels)
            if norm_layer is not None:
                self.bn = True
                self.bn_layer = norm_layer(in_channels=in_channels, **({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x, *args, **kwargs):
        batch_size, channels, height, width = x.shape
        # Embedded space by g
        g_x = self.g(x)
        if self.subsample:
            g_x = self.pool_layer(g_x)
        g_x = g_x.reshape(batch_size, self.inter_channels, -1)
        # Embedded space by phi
        phi_x = self.phi(x)
        if self.subsample:
            phi_x = self.pool_layer(phi_x)
        phi_x = phi_x.reshape(batch_size, self.inter_channels, -1)
        # Embedded space by theta
        theta_x = self.theta(x).reshape(batch_size, self.inter_channels, -1).transpose(axes=(0, 2, 1))

        if self.mode == 'dot_product':
            energy = F.linalg_gemm2(theta_x, phi_x)
            energy = energy / energy.shape[-1]
        else:
            energy = F.linalg_gemm2(theta_x, phi_x)
            energy = F.softmax(energy, axis=-1)

        out = F.linalg_gemm2(g_x, energy.transpose(axes=(0, 2, 1)))
        out = out.reshape(batch_size, self.inter_channels, height, width)
        out = self.z(out)
        if self.bn:
            out = self.bn_layer(out)

        return out + x


class SpatialPyramidPool(nn.HybridBlock):
    """
    Spatial Pyramid Module. Not support hybridize() function.
    Reference: He, K., Zhang, X., Ren, S., & Sun, J. (2015). Spatial pyramid pooling in deep
        convolutional networks for visual recognition.
        IEEE transactions on pattern analysis and machine intelligence, 37(9), 1904-1916.
    """

    def __init__(self, pool_size=(1, 2, 4), pool_type='max'):
        """
        initialize

        :param pool_size: tuple, default (1,2,4) where the output pool size is: 1x1, 2x2, 4x4
        :param pool_type: string, default 'max', options are: 'max','avg'
        """
        super(SpatialPyramidPool, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type

    def hybrid_forward(self, F, x, *args, **kwargs):
        out = None
        num_samples, num_channels, height, width = x.shape
        for i in range(len(self.pool_size)):
            kernel_size = (math.ceil(height / self.pool_size[i]), math.ceil(width / self.pool_size[i]))
            padding = (math.floor((kernel_size[0] * self.pool_size[i] - height + 1) / 2),
                       math.floor((kernel_size[1] * self.pool_size[i] - width + 1) / 2))
            pool_out = F.Pooling(x, kernel=kernel_size, pool_type=self.pool_type, stride=kernel_size, pad=padding)
            if i == 0:
                out = F.flatten(pool_out)
            else:
                out = F.concat(out, F.flatten(pool_out), dim=1)

        return out


class DeformableConv2D(nn.HybridBlock):
    """
    Deformable Convolution which is only implemented on GPU. Just a try and not suggest to use.
    """

    def __init__(self, in_channels, channels, kernel_size, strides=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1,
                 activation=None, use_bias=True):
        super(DeformableConv2D, self).__init__()
        kernel_size = self.to_tuple(kernel_size)
        self._kwargs = {'kernel': kernel_size, 'stride': self.to_tuple(strides), 'dilate': self.to_tuple(dilation),
                        'pad': self.to_tuple(padding), 'num_filter': channels, 'num_group': groups,
                        'no_bias': not use_bias}
        with self.name_scope():
            self.offset_conv = nn.Conv2D(channels=18, kernel_size=3, strides=1, padding=1, use_bias=False)
            self.weight = self.params.get('weight',
                                          shape=(channels, in_channels, kernel_size[0], kernel_size[1]),
                                          allow_deferred_init=False)
            if use_bias:
                self.bias = self.params.get('bias', shape=(channels,), allow_deferred_init=False)
            else:
                self.bias = None

            if activation is not None:
                self.act = nn.Activation(activation)
            else:
                self.act = None

    def hybrid_forward(self, F, x, *args, **kwargs):
        offset = self.offset_conv(x)
        # bias
        if self.bias:
            out = F.contrib.DeformableConvolution(x, offset, self.weight.data(), self.bias.data(), **self._kwargs)
        else:
            out = F.contrib.DeformableConvolution(x, offset, self.weight.data(), **self._kwargs)
        # activation
        if self.act:
            out = self.act(out)
        return out

    @staticmethod
    def to_tuple(params):
        if isinstance(params, base.numeric_types):
            params = (params,) * 2
        return params
