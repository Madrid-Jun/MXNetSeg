# coding=utf-8

from .segbase import SegBaseModel, ABC
from mxnetseg.tools import data_dir
from gluoncv.model_zoo import mobilenet_v2_0_25, mobilenet_v2_0_5, mobilenet_v2_0_75, mobilenet_v2_1_0

__all__ = ['SegBaseMobileNet']


def _get_backbone(name, **kwargs):
    """
    get model class and channels by name.
    """
    models = {
        'mobilenet_v2_0_25': (mobilenet_v2_0_25, (6, 8, 24, 80)),
        'mobilenet_v2_0_5': (mobilenet_v2_0_5, (12, 16, 48, 160)),
        'mobilenet_v2_0_75': (mobilenet_v2_0_75, (18, 24, 72, 240)),
        'mobilenet_v2_1_0': (mobilenet_v2_1_0, (24, 32, 96, 320)),
    }
    name = name.lower()
    if name not in models.keys():
        raise NotImplementedError(f"model {str(name)} not exists.")
    model_class, num_channels = models[name]
    net = model_class(**kwargs)
    return net, num_channels


class SegBaseMobileNet(SegBaseModel, ABC):
    """
    Backbone MobileNetv2.
    """

    def __init__(self, nclass: int, aux: bool, backbone: str = 'mobilenet_v2_1_0', height: int = None,
                 width: int = None, base_size: int = 520, crop_size: int = 480,
                 pretrained_base: bool = True, **kwargs):
        super(SegBaseMobileNet, self).__init__(nclass, aux, height, width, base_size, crop_size)
        pre_trained, self.base_channels = _get_backbone(backbone, pretrained=pretrained_base,
                                                        root=data_dir(), **kwargs)
        with self.name_scope():
            self.layer1 = pre_trained.features[:6]
            self.layer2 = pre_trained.features[6:9]
            self.layer3 = pre_trained.features[9:16]
            self.layer4 = pre_trained.features[16:20]

    def base_forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
