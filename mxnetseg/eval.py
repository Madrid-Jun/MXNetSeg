# coding=utf-8

import os
import sys
import argparse

# runtime environment
env_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if env_dir not in sys.path:
    sys.path.insert(0, env_dir)

from mxnetseg.engine import EvalFactory
from mxnetseg.models import get_model_by_name
from mxnetseg.data import get_dataset_info
from mxnetseg.tools import get_contexts, get_bn_layer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='eprnet',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='backbone network')
    parser.add_argument('--norm', type=str, default='bn',
                        help='norm layer')
    parser.add_argument('--dilate', action='store_true', default=False,
                        help='dilated backbone')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='auxiliary segmentation head')
    parser.add_argument('--checkpoint', type=str,
                        default='eprnet_resnet18_CamVidFull_08_19_14_52_17_best.params',
                        help='checkpoint')

    parser.add_argument('--ctx', type=int, nargs='+', default=(0,),
                        help="ids of GPUs or leave None to use CPU")

    parser.add_argument('--data', type=str, default='CamVidFull',
                        help='dataset name')
    parser.add_argument('--crop', type=int, default=768,
                        help='crop size')
    parser.add_argument('--base', type=int, default=960,
                        help='random scale base size')
    parser.add_argument('--mode', type=str, default='val',
                        choices=('val', 'testval', 'test'),
                        help='evaluation/prediction on val/test set')
    parser.add_argument('--ms', action='store_true', default=False,
                        help='enable multi-scale and flip skills')
    parser.add_argument('--save-dir', type=str,
                        default='C:\\Users\\BedDong\\Desktop\\results',
                        help='path to save predictions for windows')

    arg = parser.parse_args()
    return arg


def main():
    args = parse_args()
    ctx = get_contexts(args.ctx)

    data_dir, nclass = get_dataset_info(args.data)
    norm_layer, norm_kwargs = get_bn_layer(args.norm, ctx)
    model_kwargs = {
        'nclass': nclass,
        'backbone': args.backbone,
        'aux': args.aux,
        'base_size': args.base,
        'crop_size': args.crop,
        'norm_layer': norm_layer,
        'norm_kwargs': norm_kwargs,
        'dilate': args.dilate,
        'pretrained_base': False,
    }
    net = get_model_by_name(args.model, model_kwargs, args.checkpoint, ctx=ctx)

    EvalFactory.eval(net=net,
                     ctx=ctx,
                     data_name=args.data,
                     data_dir=data_dir,
                     mode=args.mode,
                     ms=args.ms,
                     nclass=nclass,
                     save_dir=args.save_dir)

    # EvalFactory.model_summary(net=net,
    #                           ctx=ctx,
    #                           data_size=(720, 960))
    #
    # EvalFactory.speed(net=net,
    #                   ctx=ctx,
    #                   data_size=(720, 960),
    #                   iterations=1000,
    #                   warm_up=500)


if __name__ == '__main__':
    main()
