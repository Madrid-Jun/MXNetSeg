# coding=utf-8

from .segbase import SegBaseModel, ABC
from mxnetseg.tools import data_dir
from gluoncv.model_zoo import densenet121, densenet161, densenet169, densenet201

__all__ = ['SegBaseDenseNet']


def _get_backbone(name, **kwargs):
    """
    get model class and channels by name.
    """
    models = {
        'densenet121': (densenet121, (256, 512, 1024, 1024)),
        'densenet161': (densenet161, (384, 768, 2112, 2208)),
        'densenet169': (densenet169, (256, 512, 1280, 1664)),
        'densenet201': (densenet201, (256, 512, 1792, 1920)),
    }
    name = name.lower()
    if name not in models.keys():
        raise NotImplementedError(f"model {str(name)} not exists.")
    model_class, num_channels = models[name]
    net = model_class(**kwargs)
    return net, num_channels


class SegBaseDenseNet(SegBaseModel, ABC):
    def __init__(self, nclass: int, aux: bool, backbone: str = 'densenet121', height: int = None,
                 width: int = None, base_size: int = 520, crop_size: int = 480,
                 pretrained_base: bool = True, **kwargs):
        super(SegBaseDenseNet, self).__init__(nclass, aux, height, width, base_size, crop_size)
        pre_trained, self.base_channels = _get_backbone(backbone, pretrained=pretrained_base,
                                                        root=data_dir(), **kwargs)
        with self.name_scope():
            self.stem = pre_trained.features[0:4]
            self.layer1 = pre_trained.features[4:5]
            self.layer2 = pre_trained.features[5:7]
            self.layer3 = pre_trained.features[7:9]
            self.layer4 = pre_trained.features[9:13]

    def base_forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4
