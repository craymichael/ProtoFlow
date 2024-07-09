"""
GNU GPL v2.0
Copyright (c) 2024 Zachariah Carmichael, Timothy Redgrave, Daniel Gonzalez Cedre
ProtoFlow Project

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import os
import os.path as osp

from protoflow.utils import profile

LOG_DIR = os.getenv('LOG_DIR', 'logs')


def setup(debug):
    import torch

    if debug:
        rank = local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)
    else:
        import torch.distributed as dist

        try:
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ['WORLD_SIZE'])

            torch.cuda.set_device(local_rank)
            dist.init_process_group('nccl')
        except KeyError:
            rank = local_rank = 0
            world_size = 1

            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://127.0.0.1:12584',
                rank=rank,
                world_size=world_size,
            )

    return rank, world_size


def cleanup():
    import torch.distributed as dist

    dist.destroy_process_group()


def now_str():
    from datetime import datetime

    return datetime.now().isoformat(timespec='seconds').replace(':', '_')


@profile
def run(args):
    rank, world_size = setup(debug=False)

    try:
        import pickle
        import json
        from copy import deepcopy

        import torch

        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.optim.swa_utils import update_bn

        from experiments.image.model.model_flow import get_model
        from protoflow.proto import ProtoFlowGMM
        from protoflow.datasets import get_dataset
        from protoflow.training import get_transform
        from protoflow.evaluation import test
        from protoflow.utils import dict_to_namespace
        from protoflow.utils import convert_legacy_config
        from protoflow.utils_heavy import make_dataloader

        torch.backends.cudnn.benchmark = True

        run_folder = osp.dirname(args.resume)
        if rank == 0:
            print(f'Run folder: {run_folder}')

        config_path = osp.join(run_folder, 'config.json')
        if osp.exists(config_path):
            with open(config_path, 'r') as fp:
                config = json.load(fp)
            config = convert_legacy_config(config)
            flow_args = dict_to_namespace(config['flow_args'])
            interpolation = config['interpolation']
            img_size = config['img_size']
            augmentation = config.get('augmentation', args.augmentation)
            protos_per_class = config.get('protos_per_class', 10)
            gaussian_approach = config.get('gaussian_approach',
                                           'GaussianMixture')
            use_base_dist = config.get('elbo_loss', False)
            likelihood_approach = config.get('likelihood_approach', 'total')
        else:
            if rank == 0:
                print(f'WARNING: no config file found at {config_path}')
            if args.flow_ckpt is None:
                raise RuntimeError('You must specify --flow_ckpt if no config '
                                   'file is found!')

            path_args = '{}/args.pickle'.format(args.flow_ckpt)
            with open(path_args, 'rb') as f:
                flow_args = pickle.load(f)

            if args.interpolation is None:
                print('The option --interpolation should be specified if no '
                      'config file is found! Assuming bicubic...')
                interpolation = 'bicubic'
            else:
                interpolation = args.interpolation

            if args.img_size is None:
                raise RuntimeError('You must specify --img_size if no config '
                                   'file is found!')
            config = {}
            img_size = args.img_size
            augmentation = args.augmentation
            protos_per_class = 10
            gaussian_approach = 'GaussianMixture'
            use_base_dist = True
            likelihood_approach = 'total'

        n_channels = 3
        flow_model = get_model(flow_args,
                               data_shape=(n_channels, img_size, img_size),
                               base_dist=use_base_dist)
        flow_model.to(rank)

        n_classes = {
            'cub200': 200,
            'cifar10': 10,
            'mnist': 10,
            'cifar100': 100,
            'pets': 37,
            'flowers': 102,
            'imagenet': 1000,
        }[args.dataset]

        model = ProtoFlowGMM(
            model=flow_model,
            n_classes=n_classes,
            features_shape=flow_model.out_shape,
            protos_per_class=protos_per_class,
            gaussian_approach=gaussian_approach,

            likelihood_approach=args.likelihood_approach or likelihood_approach,
        )
        model.to(rank)
        model.eval()

        model = DDP(model, device_ids=[rank])

        if rank == 0:
            print(f'Loading checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
        model.module.load_state_dict(checkpoint['model'])

        if args.var_temp is not None:
            for gmm in model.module.gmms:
                gmm.var.data = gmm.var * args.var_temp

        multi_transform_k = args.tta_num if args.tta else None
        transform = get_transform(interpolation, img_size, train=args.tta,
                                  augmentation=augmentation,
                                  ten_crop=args.ten_crop,
                                  multi_transform_k=multi_transform_k)

        dl_train = None
        if not args.test_only:
            ds_train = get_dataset(args.dataset, train=True,
                                   transform=transform)
            dl_train = make_dataloader(ds_train, rank, world_size,
                                       args.batch_size)
        ds_test = get_dataset(args.dataset, train=False, transform=transform)
        dl_test = make_dataloader(ds_test, rank, world_size, args.batch_size)

        to_eval = [('raw', model)]

        if config.get('use_ema', False):
            model_ema = deepcopy(model)
            model_ema.module.load_state_dict(checkpoint['model_ema'])
            if not (checkpoint['model_ema_updated_bn'] or args.no_ema_stats):
                if rank == 0:
                    print('Updating batch norm stats for the EMA model')

                train_transform = get_transform(
                    interpolation, img_size, train=True,
                    augmentation=augmentation,
                    ten_crop=False, multi_transform_k=None,
                )
                ds_train_plain = get_dataset(args.dataset, train=True,
                                             transform=train_transform)
                dl_train_plain = make_dataloader(
                    ds_train_plain, rank, world_size, args.batch_size,
                    persistent_workers=False
                )

                update_bn(dl_train_plain, model_ema, device=rank)
                del dl_train_plain

            to_eval.append(('EMA', model_ema))

        for model_name, the_model in to_eval:
            train_scores = None
            if not args.test_only:
                if rank == 0:
                    print(f'Evaluating {model_name} model on train set...')
                train_scores = test(
                    rank=rank,
                    model=the_model,
                    dl=dl_train,
                    n_classes=n_classes,
                    num_samples=args.num_samples,
                    ten_crop=args.ten_crop,
                    multi_transform_k=multi_transform_k,
                    calibration_metrics=True,
                    args=args,
                )
                if rank == 0:
                    print(f'EVALUATION STATS (train, {model_name}):')
                    for name, score in train_scores.items():
                        print(f'  {name}: {score.item():.3f}')

            if rank == 0:
                print(f'Evaluating {model_name} model on test set...')
            test_scores = test(
                rank=rank,
                model=the_model,
                dl=dl_test,
                n_classes=n_classes,
                num_samples=args.num_samples,
                ten_crop=args.ten_crop,
                multi_transform_k=multi_transform_k,
                calibration_metrics=True,
                args=args,
            )
            if rank == 0:
                print(f'EVALUATION STATS (test, {model_name}):')
                for name, score in test_scores.items():
                    print(f'  {name}: {score.item():.3f}')

                result_dir = osp.join(run_folder, f'scores_{model_name}')
                os.makedirs(result_dir, exist_ok=True)
                write_path = osp.join(result_dir, f'scores_{now_str()}.json')
                print(f'Write results to {write_path}')
                score_data = {
                    'dataset': args.dataset,
                    'resume': args.resume,
                    'num_samples': args.num_samples,
                    'ten_crop': args.ten_crop,
                    'var_temp': args.var_temp,
                    'likelihood_approach': args.likelihood_approach,
                    'tta': args.tta,
                    'tta_num': args.tta_num if args.tta else 0,
                    'scores_test': {
                        k: v.item() for k, v in test_scores.items()},
                }
                if not args.test_only:
                    score_data['scores_train'] = {
                        k: v.item() for k, v in train_scores.items()}
                with open(write_path, 'w') as fp:
                    json.dump(score_data, fp, indent=2)
    finally:
        cleanup()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--resume', type=str, required=True,
                        help='Path to resume checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10',
                                 'food', 'caltech101', 'imagenet', 'cifar100',
                                 'objectnet', 'aircraft', 'cub200'],
                        help='Dataset to use')
    parser.add_argument('--batch_size', '-b', type=int, default=2048)
    parser.add_argument('--batch_steps', '-s', type=int, default=1)
    parser.add_argument('--test_only', action='store_true',
                        help='Only evaluation on the test split')
    parser.add_argument('--no_ema_stats', action='store_true',
                        help='If applicable, do not compute EMA statistics')
    parser.add_argument('--num_samples', '-n', type=int, default=1,
                        help='Number of monte carlo samples for DenseFlow model')
    parser.add_argument('--ten_crop', action='store_true',
                        help='Evaluate on 10 crops of the same image')
    parser.add_argument('--tta', action='store_true',
                        help='Use test time augmentation')
    parser.add_argument('--tta_num', type=int, default=5,
                        help='Number of test time augmentations to use')
    parser.add_argument('--var_temp', '-T', type=float, default=None,
                        help='Variance temperature')
    parser.add_argument('--likelihood_approach', default=None,
                        choices=('total', 'max'),
                        help='GMM likelihood approach per class')

    legacy = parser.add_argument_group('Legacy (No Config File)')
    legacy.add_argument('--flow_ckpt', type=str, default=None)
    legacy.add_argument('--augmentation', default='v1')
    legacy.add_argument('--interpolation', type=str, default=None,
                        help='Resize interpolation type')
    legacy.add_argument('--img_size', type=int, default=None,
                        help='Image size')

    args = parser.parse_args()
    print(args)

    run(args)


if __name__ == '__main__':
    main()
