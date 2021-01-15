# coding=utf-8

import os
import numpy as np

from gluoncv.data import COCOSegmentation, ADE20KSegmentation
from .voc import VOCSegmentation
from .sbd import SBDSegmentation
from .vocaug import VOCAugSegmentation
from .pcontext import PContextSegmentation
from .sunrgbd import SunRGBDSegmentation
from .nyu import NYUSegmentation
from .sift import SiftFlowSegmentation
from .stanford import StanfordSegmentation
from .aeroscapes import AeroSegmentation
from .cityscapes import CitySegmentation
from .camvid import CamVidSegmentation
from .gatech import GATECHSegmentation
from .mapillary import MapillarySegmentation
from .bdd import BDDSegmentation
from .kitti import KITTIZhSementation, KITTIXuSementation, KITTIRosSementation
from .mhp import MHPV1Segmentation

from mxnetseg.tools import dataset_dir

__all__ = ['segmentation_dataset', 'get_dataset_info', 'verify_classes']

_data_sets = {
    'ade20k': (ADE20KSegmentation, 'ADE20K'),
    'coco': (COCOSegmentation, 'COCO'),
    'voc2012': (VOCSegmentation, 'VOCdevkit'),
    'sbd': (SBDSegmentation, 'VOCdevkit'),
    'vocaug': (VOCAugSegmentation, 'VOCdevkit'),
    'pcontext': (PContextSegmentation, 'PContext'),
    'sunrgbd': (SunRGBDSegmentation, 'SUNRGBD'),
    'nyu': (NYUSegmentation, 'NYU'),
    'siftflow': (SiftFlowSegmentation, 'SiftFlow'),
    'stanford': (StanfordSegmentation, 'Stanford10'),
    'aeroscapes': (AeroSegmentation, 'Aeroscapes'),
    'cityscapes': (CitySegmentation, 'Cityscapes'),
    'camvid': (CamVidSegmentation, 'CamVid'),
    'camvidfull': (CamVidSegmentation, 'CamVidFull'),
    'gatech': (GATECHSegmentation, 'GATECH'),
    'mapillary': (MapillarySegmentation, 'Mapillary'),
    'bdd': (BDDSegmentation, 'BDD'),
    'kittizhang': (KITTIZhSementation, 'KITTI'),
    'kittixu': (KITTIXuSementation, 'KITTI'),
    'kittiros': (KITTIRosSementation, 'KITTI'),
    'mhpv1': (MHPV1Segmentation, 'MHP'),

}


def segmentation_dataset(name: str, **kwargs):
    """
    corresponding segmentation dataset

    :param name: dataset name
    :param kwargs: keywords depending on the API
    """
    dataset, _ = _data_sets[name.lower()]
    return dataset(**kwargs)


def get_dataset_info(name: str):
    """
    basic dataset information
    :param name: dataset name
    :return: dataset dir and number of classes
    """
    dataset, folder_name = _data_sets[name.lower()]
    return os.path.join(dataset_dir(), folder_name), dataset.NUM_CLASS


def verify_classes(name, split='train'):
    """ to verify class labels """
    dataset, folder_name = _data_sets[name.lower()]
    data_dir = os.path.join(dataset_dir(), folder_name)
    # train set
    if split == 'train':
        train_set = dataset(data_dir, split='train')
        print("Train images: %d" % len(train_set))
        train_class_num = _verify_classes_assist([train_set])
        print("Train class label is: %s" % str(train_class_num))
    # val set
    elif split == 'val':
        val_set = dataset(data_dir, split='val')
        print("Val images: %d" % len(val_set))
        val_class_num = _verify_classes_assist([val_set])
        print("Val class label is: %s" % str(val_class_num))
    else:
        raise RuntimeError(f"Unknown split: {split}")


def _verify_classes_assist(data_list):
    from tqdm import tqdm
    uniques = []
    for datas in data_list:
        tbar = tqdm(datas)
        for _, (_, mask) in enumerate(tbar):
            mask = mask.asnumpy()
            unique = np.unique(mask)
            for v in unique:
                if v not in uniques:
                    uniques.append(v)
    uniques.sort()
    return uniques
