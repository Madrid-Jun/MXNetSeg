# coding=utf-8


import os
import numpy as np
import mxnet as mx
from PIL import Image
from gluoncv.data.segbase import SegmentationDataset


class KITTIZhSementation(SegmentationDataset):
    """ KITTI Segmentation Dataset annotated by Richard Zhang:
    Richard Zhang, et al. Sensor fusion for semantic segmentation of urban scenes. In ICRA 2015. """

    NUM_CLASS = 10
    BASE_DIR = 'Zhang'

    def __init__(self, root=os.path.expanduser('~/.mxnet/dataset/KITTI'), split='train',
                 mode=None, transform=None, **kwargs):
        super(KITTIZhSementation, self).__init__(root, split, mode, transform, **kwargs)
        _kitti_root = os.path.join(root, self.BASE_DIR)
        _img_dir = os.path.join(_kitti_root, 'images')
        _mask_dir = os.path.join(_kitti_root, 'labels')
        if split == 'train':
            _split_f = os.path.join(_kitti_root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_kitti_root, 'test.txt')
        elif split == 'test':
            _split_f = os.path.join(_kitti_root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _file_path = line.strip()

                _image = os.path.join(_img_dir, _file_path)
                assert os.path.isfile(_image)
                self.images.append(_image)

                _mask = os.path.join(_mask_dir, _file_path)
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            img_path = self.images[idx]
            # return img, os.path.basename(self.images[idx])
            return img, img_path[img_path.index('images') + len('images') + 1:]
        mask = Image.open(self.masks[idx])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('void', 'building', 'sky', 'road', 'vegetation', 'sidewalk',
                'car', 'pedestrian', 'cyclist', 'signage', 'fence')


class KITTIXuSementation(SegmentationDataset):
    """ KITTI Segmentation Dataset annotated by Xu.
    Ph. Xu et al. Multimodal Information Fusion for Urban Scene Understanding. In MVA 2016. """

    NUM_CLASS = 28
    BASE_DIR = 'Xu'

    def __init__(self, root=os.path.expanduser('~/.mxnet/dataset/KITTI'), split='train',
                 mode=None, transform=None, **kwargs):
        super(KITTIXuSementation, self).__init__(root, split, mode, transform, **kwargs)
        _kitti_root = os.path.join(root, self.BASE_DIR)
        _img_dir = os.path.join(_kitti_root, 'images')
        _mask_dir = os.path.join(_kitti_root, 'labels')
        if split == 'train':
            _split_f = os.path.join(_kitti_root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_kitti_root, 'test.txt')
        elif split == 'test':
            _split_f = os.path.join(_kitti_root, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _file_path = line.strip()

                _image = os.path.join(_img_dir, _file_path)
                assert os.path.isfile(_image)
                self.images.append(_image)

                _mask = os.path.join(_mask_dir, _file_path)
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[idx])
        mask = Image.open(self.masks[idx])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32') - 1
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('void', 'sky', 'road', 'sidewalk', 'bicyclelane', 'lanemarkers', 'railway',
                'grass', 'tree', 'othervege', 'building', 'wall', 'bridge', 'pole',
                'panel', 'fence', 'trafficlight', 'otherinf', 'car', 'truck', 'train',
                'bus', 'adult', 'children', 'cyclist', 'bicycle', 'motocyclist',
                'motocycle', 'animal', 'othermov')


class KITTIRosSementation(SegmentationDataset):
    """ KITTI Segmentation Dataset annotated by G. Ros.
    G. Ros et al. Vision-based Offline-Online Perception Paradigm for Autonomous Driving. In WACV 2015. """

    NUM_CLASS = 11
    BASE_DIR = 'Ros'

    def __init__(self, root=os.path.expanduser('~/.mxnet/dataset/KITTI'), split='train',
                 mode=None, transform=None, **kwargs):
        super(KITTIRosSementation, self).__init__(root, split, mode, transform, **kwargs)
        _kitti_root = os.path.join(root, self.BASE_DIR)
        if split == 'train':
            _split_f = os.path.join(_kitti_root, 'Training_00')
        elif split == 'val':
            _split_f = os.path.join(_kitti_root, 'Validation_07')
        elif split == 'test':
            _split_f = os.path.join(_kitti_root, 'Validation_07')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))

        self.images = []
        self.masks = []
        for _, _, files in os.walk(os.path.join(_split_f, 'RGB')):
            for file in files:
                _image = os.path.join(_split_f, 'RGB', file)
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_split_f, 'GT_GRAY', file)
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[idx])
        mask = Image.open(self.masks[idx])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 11] = -1
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return ('sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation',
                'pole', 'car', 'sign', 'pedestrian', 'cyclist', 'void')
