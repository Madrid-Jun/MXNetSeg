# coding=utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import axis, title
from mxnet import nd
from typing import Tuple, Union


def get_row_col(num_pic):
    """
    get figure row and column number
    """
    sqr = num_pic ** 0.5
    row = round(sqr)
    col = row + 1 if sqr - row > 0 else row
    return row, col


def format_to_plot(tensor: nd.NDArray) -> np.ndarray:
    """format the input tensor from NCHW/CHW to HWC/HW for plotting"""
    if len(tensor.shape) == 4:
        tensor = nd.squeeze(tensor, axis=0)
    if tensor.shape[0] == 1:
        tensor = nd.squeeze(tensor, axis=0)
    else:
        tensor = nd.transpose(tensor, axes=(1, 2, 0))
    return tensor.asnumpy()


def plot_features(features: nd.NDArray, scope: int = 9):
    """
    visualize feature maps per channel.

    :param features: feature map with shape 1xCxHxW
    :param scope: the index of feature maps to visualize is [0, scope)
    """
    scope = scope if scope < features.shape[1] else features.shape[1]
    feature_maps = nd.squeeze(features, axis=0).asnumpy()
    feature_map_combination = []
    # separate visualization
    row, col = get_row_col(scope)
    plt.figure()
    for i in range(0, scope):
        feature_map = feature_maps[i, :, :]
        feature_map_combination.append(feature_map)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map)
        axis('off')
        title(f"feature map {i}")
    plt.show()
    # overlaps
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.show()


def overlap_heat_map(img_path: str, heat_map: Union[np.ndarray, nd.NDArray],
                     alpha: float = 0.4, save_name=None):
    """
    overlap the original image and heat-map.
    """
    import cv2
    if isinstance(heat_map, nd.NDArray):
        heat_map = heat_map.asnumpy()
    assert len(heat_map.shape) == 2
    img = cv2.imread(img_path)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_GRAY2BGR)
    heat_map = cv2.resize(heat_map, (img.shape[1], img.shape[0]))
    heat_map = np.uint8(255 * heat_map)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    overlap = heat_map * alpha + img
    if save_name:
        cv2.imwrite(os.path.join(os.path.dirname(img_path), save_name + '.png'), overlap)
    return overlap


def plot_heat_map(heat_map: Union[nd.NDArray, np.ndarray], sns: bool = False, fig_width: int = 16,
                  save_pth=None):
    """
    plot heat-map and keep height-width ratio
    :param heat_map: expect input shape of HxW
    :param sns: use seaborn package
    :param fig_width: width of figure
    :param save_pth: save figure when not none
    """
    assert len(heat_map.shape) == 2
    if isinstance(heat_map, nd.NDArray):
        heat_map = heat_map.asnumpy()
    h, w = heat_map.shape
    ratio = h / w
    plt.figure(figsize=(fig_width, int(fig_width * ratio)))
    if sns:
        import seaborn
        seaborn.set()
        seaborn.heatmap(heat_map, cbar=True)
    else:
        plt.imshow(heat_map, cmap=plt.cm.hot)
    plt.axis('off')
    if save_pth:
        dir_path, _ = os.path.split(save_pth)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(save_pth)
        plt.close()
    else:
        plt.show()


def plot_pixel(img_path: str, position: Tuple[int, int]):
    """
    mark a specific pixel with cross sign.
    :param img_path: image path
    :param position: the pixel coordinates
    """
    pass
