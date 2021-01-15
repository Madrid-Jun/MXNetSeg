# coding=utf-8

import json
import numpy as np
from PIL import Image
from gluoncv.utils.viz import get_color_pallete
from .path import mapillary_config

__all__ = ['my_color_palette', 'city_train2label']


def city_train2label(npimg):
    """convert train ID to label ID for cityscapes"""
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                     23, 24, 25, 26, 27, 28, 31, 32, 33]
    pred = np.array(npimg, np.uint8)
    label = np.zeros(pred.shape)
    ids = np.unique(pred)
    for i in ids:
        label[np.where(pred == i)] = valid_classes[i]
    out_img = Image.fromarray(label.astype('uint8'))
    return out_img


def my_color_palette(npimg, dataset: str):
    """
    Visualize image and return PIL.Image with color palette.

    :param npimg: Single channel numpy image with shape `H, W, 1`
    :param dataset: dataset name
    """

    if dataset in ('pascal_voc', 'ade20k', 'citys', 'mhpv1'):
        return get_color_pallete(npimg, dataset=dataset)

    elif dataset == 'camvid':
        npimg[npimg == -1] = 11
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(cam_palette)
        return out_img
    elif dataset == 'mapillary':
        npimg[npimg == -1] = 65
        color_img = _apply_mapillary_palette(npimg)
        out_img = Image.fromarray(color_img)
        return out_img
    elif dataset == 'aeroscapes':
        out_img = Image.fromarray(npimg.astype('uint8'))
        out_img.putpalette(aeroscapes_palette)
        return out_img
    else:
        raise RuntimeError("Un-defined palette for data {}".format(dataset))


def _apply_mapillary_palette(image_array):
    with open(mapillary_config()) as config_file:
        config = json.load(config_file)
    labels = config['labels']
    # palette
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
    for label_id, label in enumerate(labels):
        color_array[image_array == label_id] = label["color"]
    return color_array


cam_palette = [
    128, 128, 128,  # sky
    128, 0, 0,  # building
    192, 192, 128,  # column_pole
    128, 64, 128,  # road
    0, 0, 192,  # sidewalk
    128, 128, 0,  # tree
    192, 128, 128,  # SignSymbol
    64, 64, 128,  # fence
    64, 0, 128,  # car
    64, 64, 0,  # pedestrian
    0, 128, 192,  # bicyclist
    0, 0, 0  # void
]

aeroscapes_palette = [
    0, 0, 0,  # background
    192, 128, 128,  # person
    0, 128, 0,  # bike
    128, 128, 128,  # car
    128, 0, 0,  # drone
    0, 0, 128,  # boat
    192, 0, 128,  # animal
    192, 0, 0,  # obstacle
    192, 128, 0,  # construction
    0, 64, 0,  # vegetation
    128, 128, 0,  # road
    0, 128, 128  # sky
]
