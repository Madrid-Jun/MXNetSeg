# coding=utf-8

import platform

from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader
from gluoncv.utils import LRScheduler

from mxnetseg.models import get_model_by_name
from mxnetseg.data import get_dataset_info, segmentation_dataset
from mxnetseg.tools import get_bn_layer, image_transform


class FitFactory:
    """
    methods for model training
    """

    @staticmethod
    def get_model(cfg: dict, ctx: list):
        norm_layer, norm_kwargs = get_bn_layer(cfg.get('norm'), ctx)
        model_kwargs = {
            'nclass': get_dataset_info(cfg.get('data_name'))[1],
            'backbone': cfg.get('backbone'),
            'pretrained_base': cfg.get('backbone_pretrain'),
            'aux': cfg.get('aux'),
            'crop_size': cfg.get('crop_size'),
            'base_size': cfg.get('base_size'),
            'dilate': cfg.get('dilate'),
            'norm_layer': norm_layer,
            'norm_kwargs': norm_kwargs,
        }
        model = get_model_by_name(name=cfg.get('model_name'),
                                  model_kwargs=model_kwargs,
                                  resume=cfg.get('resume'),
                                  lr_mult=cfg.get('lr_mult'),
                                  ctx=ctx)
        model.hybridize()
        return model

    @staticmethod
    def data_iter(data_name, batch_size, shuffle=True, last_batch='discard', **kwargs):
        transform = image_transform()
        num_worker = 0 if platform.system() == 'Windows' else 16
        data_set = segmentation_dataset(data_name, transform=transform, **kwargs)
        data_iter = DataLoader(data_set, batch_size, shuffle,
                               last_batch=last_batch, num_workers=num_worker)
        return data_iter, len(data_set)

    @staticmethod
    def get_criterion(aux, aux_weight, focal_kwargs=None, sensitive_kwargs=None, ohem=False):
        if focal_kwargs:
            from mxnetseg.nn import FocalLoss
            return FocalLoss(**focal_kwargs)
        if sensitive_kwargs:
            raise NotImplementedError
        if ohem:
            from gluoncv.loss import MixSoftmaxCrossEntropyOHEMLoss
            return MixSoftmaxCrossEntropyOHEMLoss(aux, aux_weight)
        else:
            from gluoncv.loss import MixSoftmaxCrossEntropyLoss
            # from mxnetseg.nn import MixSoftmaxCrossEntropyLoss
            return MixSoftmaxCrossEntropyLoss(aux, aux_weight=aux_weight)

    @staticmethod
    def lr_scheduler(mode, base_lr, target_lr, nepochs, iters_per_epoch,
                     step_epoch=None, step_factor=0.1, power=0.9):
        assert mode in ('constant', 'step', 'linear', 'poly', 'cosine')
        sched_kwargs = {
            'base_lr': base_lr,
            'target_lr': target_lr,
            'nepochs': nepochs,
            'iters_per_epoch': iters_per_epoch,
        }
        if mode == 'step':
            sched_kwargs['mode'] = 'step'
            sched_kwargs['step_epoch'] = step_epoch
            sched_kwargs['step_factor'] = step_factor
        elif mode == 'poly':
            sched_kwargs['mode'] = 'poly'
            sched_kwargs['power'] = power
        else:
            sched_kwargs['mode'] = mode
        return LRScheduler(**sched_kwargs)

    @staticmethod
    def create_trainer(model=None, cfg=None, iters_per_epoch=0):
        if cfg.get('optimizer') == 'adam':
            trainer = Trainer(model.collect_params(), 'adam',
                              optimizer_params={'learning_rate': cfg.get('lr'),
                                                'wd': cfg.get('wd'),
                                                'beta1': cfg.get('adam').get('adam_beta1'),
                                                'beta2': cfg.get('adam').get('adam_beta2')})
        elif cfg.get('optimizer') in ('sgd', 'nag'):
            scheduler = FitFactory.lr_scheduler(mode=cfg.get('lr_scheduler'),
                                                base_lr=cfg.get('lr'),
                                                target_lr=cfg.get('target_lr'),
                                                nepochs=cfg.get('epochs'),
                                                iters_per_epoch=iters_per_epoch,
                                                step_epoch=cfg.get('step').get('step_epoch'),
                                                step_factor=cfg.get('step').get('step_factor'),
                                                power=cfg.get('poly').get('power'))
            trainer = Trainer(model.collect_params(), cfg.get('optimizer'),
                              optimizer_params={'lr_scheduler': scheduler,
                                                'wd': cfg.get('wd'),
                                                'momentum': cfg.get('momentum'),
                                                'multi_precision': True})
        else:
            raise RuntimeError(f"Unknown optimizer: {cfg.get('optimizer')}")
        return trainer
