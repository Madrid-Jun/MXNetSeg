# coding=utf-8

import os
import time
import platform
import numpy as np
from tqdm import tqdm

from mxnet import nd, autograd
from mxnet.log import get_logger
from mxnet.gluon.data import DataLoader
from gluoncv.utils.metrics import SegmentationMetric
from gluoncv.model_zoo.segbase import MultiEvalModel

from mxnetseg.data import segmentation_dataset
from mxnetseg.tools import image_transform, root_dir, city_train2label, my_color_palette

color_palette = {
    # dataset name: dataset palette key
    'voc2012': 'pascal_voc',
    'ade20k': 'ade20k',
    'cityscapes': 'citys',
    'bdd': 'citys',  # BDD100k shares the same 19 semantic categories as Cityscapes
    'mhpv1': 'mhpv1',
    'camvid': 'camvid',
    'camvidfull': 'camvid',
    'mapillarry': 'mapillarry',
}
logger = get_logger(name='eval', level=20)


class EvalFactory:
    """
    methods for model evaluation
    """

    @staticmethod
    def _sample(shape, ctx) -> nd.NDArray:
        if isinstance(shape, (list, tuple)):
            h = shape[0]
            w = shape[1]
        else:
            h = shape
            w = shape
        sample = nd.random.uniform(shape=(1, 3, h, w), ctx=ctx)
        return sample

    @staticmethod
    def speed(net, ctx, data_size=(1024, 1024), iterations=1000, warm_up=500):
        net.hybridize(static_alloc=True)
        sample = EvalFactory._sample(data_size, ctx[0])

        logger.info(f'Warm-up starts for {warm_up} forward passes...')
        for _ in range(warm_up):
            with autograd.record(False):
                net.predict(sample)
        nd.waitall()

        logger.info(f'Evaluate inference speed for {iterations} forward passes...')
        start = time.time()
        for _ in range(iterations):
            with autograd.record(False):
                net.predict(sample)
        nd.waitall()
        time_cost = time.time() - start

        logger.info('Total time: %.2fs, latency: %.2fms, FPS: %.1f'
                    % (time_cost, time_cost / iterations * 1000, iterations / time_cost))

    @staticmethod
    def model_summary(net, ctx, data_size=(1024, 1024)):
        sample = EvalFactory._sample(data_size, ctx[0])
        net.summary(sample)

    @staticmethod
    def _data_set(data_name, mode, data_dir):
        transform = image_transform()
        if mode == 'test':
            dataset = segmentation_dataset(data_name, root=data_dir, split='test', mode='test',
                                           transform=transform)
        elif mode == 'testval':
            dataset = segmentation_dataset(data_name, root=data_dir, split='val', mode='test',
                                           transform=transform)
        elif mode == 'val':
            dataset = segmentation_dataset(data_name, root=data_dir, split='val', mode='testval',
                                           transform=transform)
        else:
            raise RuntimeError(f"Unknown mode: {mode}")
        return dataset

    @staticmethod
    def _scales(ms, data_name):
        if ms:
            if data_name.lower() in ('cityscapes', 'camvid', 'bdd'):
                scales = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25)
            elif data_name.lower() in ('gatech',):
                scales = (0.5, 0.8, 1.0, 1.2, 1.4)
            else:
                scales = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
        else:
            scales = (1.0,)
        return scales

    @staticmethod
    def _prediction_dir(save_dir, data_name):
        if platform.system() == 'Linux':
            save_dir = os.path.join(root_dir(), 'color_results')
        save_dir = os.path.join(save_dir, data_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    @staticmethod
    def _colored_predictions(evaluator, dataset, data_name, save_dir):
        save_dir = EvalFactory._prediction_dir(save_dir, data_name)
        palette_key = color_palette[data_name.lower()]

        bar = tqdm(dataset)
        for _, (img, dst) in enumerate(bar):
            img = img.expand_dims(0)
            output = evaluator.parallel_forward(img)[0]
            mask = EvalFactory._mask(output, palette_key)
            save_name = dst.split('.')[0] + '.png'
            save_path = os.path.join(save_dir, save_name)
            dir_path, _ = os.path.split(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            mask.save(save_path)

    @staticmethod
    def _mask(output, palette_key, test_citys=False):
        predict = nd.squeeze(nd.argmax(output, 1)).asnumpy()
        if test_citys:
            mask = city_train2label(predict)
        else:
            mask = my_color_palette(predict, palette_key)
        return mask

    @staticmethod
    def _eval_scores(evaluator, dataset, nclass):
        metric = SegmentationMetric(nclass)
        val_iter = DataLoader(dataset, batch_size=1, last_batch='keep')
        bar = tqdm(val_iter)
        for _, (img, label) in enumerate(bar):
            pred = evaluator.parallel_forward(img)[0]
            metric.update(label, pred)
            bar.set_description("PA: %.4f    mIoU: %.4f" % metric.get())
        pix_acc, miou = metric.get()
        total_inter = metric.total_inter
        total_union = metric.total_union
        per_class_iou = 1.0 * total_inter / (np.spacing(1) + total_union)
        return pix_acc, miou, per_class_iou

    @staticmethod
    def eval(net, ctx, data_name, data_dir, mode, ms, nclass, save_dir):
        net.hybridize()
        scales = EvalFactory._scales(ms, data_name)
        dataset = EvalFactory._data_set(data_name, mode, data_dir)
        evaluator = MultiEvalModel(net, nclass, ctx, flip=ms, scales=scales)

        logger.info(f'Input scales: {scales}')

        if 'test' in mode:
            assert data_name.lower() in color_palette.keys()
            EvalFactory._colored_predictions(evaluator, dataset, data_name, save_dir)
        else:
            pa, miou, per_class_iou = EvalFactory._eval_scores(evaluator, dataset, nclass)
            for i, score in enumerate(per_class_iou):
                logger.info('class {0:2} ==> IoU {1:2}'.format(i, round(score * 100, 2)))
            logger.info("PA: %.2f, mIoU: %.2f" % (pa * 100, miou * 100))
