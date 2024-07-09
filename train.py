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
from contextlib import nullcontext

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


def cleanup(debug):
    from protoflow.datasets import cleanup_shm_data

    if not debug:
        import torch.distributed as dist

        dist.destroy_process_group()

    cleanup_shm_data()


@profile
def run(args):
    rank, world_size = setup(args.debug)
    args.world_size = world_size
    try:
        import pickle
        import json

        import git

        import torch
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data import random_split
        from torch.utils.data import Subset

        from experiments.image.model.model_flow import get_model
        from experiments.image.utils import set_seeds
        from protoflow.proto import ProtoFlowGMM
        from protoflow.datasets import get_dataset
        from protoflow.training import get_transform
        from protoflow.training import train_loop
        from protoflow.training import init_gmms
        from protoflow.evaluation import test
        from protoflow.utils_heavy import make_dataloader

        torch.backends.cudnn.benchmark = True
        set_seeds(seed=0xFACE)

        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha
        except git.InvalidGitRepositoryError:
            sha = None
        args.git_commit_hash = sha

        if args.detect_autograd_anomaly:
            torch.autograd.set_detect_anomaly(True)

        name = f'bs{args.batch_size}_{args.interpolation}_{args.img_size}'
        if args.extra is not None:
            run_folder = osp.join(LOG_DIR, args.dataset + '_' + args.extra,
                                  name)
        else:
            run_folder = osp.join(LOG_DIR, args.dataset, name)

        path_args = '{}/args.pickle'.format(args.flow_ckpt)
        with open(path_args, 'rb') as f:
            flow_args = pickle.load(f)

        if rank == 0:
            os.makedirs(run_folder, exist_ok=True)
            print(f'Run folder: {run_folder}')

            config_path = osp.join(run_folder, 'config.json')
            if osp.exists(config_path):
                with open(config_path, 'r') as fp:
                    config = json.load(fp)
                for k, v in config.items():
                    if k == 'flow_args':

                        for kk, vv in v.items():
                            assert hasattr(flow_args, kk), (kk, vv)
                            assert getattr(flow_args, kk) == vv, (
                                kk, getattr(flow_args, kk), vv)
                    else:
                        assert hasattr(args, k), (k, v)
                        assert getattr(args, k) == v, (k, getattr(args, k), v)
            else:
                if args.resume:
                    print(
                        f'WARNING: no config file found at {config_path} even '
                        f'though we are resuming training! Creating one now '
                        f'based on your arguments.')
                with open(config_path, 'w') as fp:
                    json.dump({
                        'flow_args': vars(flow_args),
                        'no_restore_flow_ckpt': args.no_restore_flow_ckpt,
                        'img_size': args.img_size,
                        'batch_size': args.batch_size,
                        'world_size': args.world_size,
                        'num_epochs': args.num_epochs,
                        'lr': args.lr,
                        'gmm_lr': args.gmm_lr,
                        'use_ema': args.use_ema,
                        'warmup_epochs': args.warmup_epochs,
                        'interpolation': args.interpolation,
                        'trainable': args.trainable,
                        'gmm_em': args.gmm_em,
                        'init_gmm': args.init_gmm,
                        'clip_grad_norm': args.clip_grad_norm,
                        'weight_decay': args.weight_decay,
                        'protos_per_class': args.protos_per_class,
                        'gaussian_approach': args.gaussian_approach,
                        'likelihood_approach': args.likelihood_approach,
                        'proto_dropout_prob': args.proto_dropout_prob,
                        'z_dropout_prob': args.z_dropout_prob,
                        'mu_loss': args.mu_loss,
                        'augmentation': args.augmentation,
                        'proto_loss': args.proto_loss,
                        'elbo_loss': args.elbo_loss,
                        'elbo_loss2': args.elbo_loss2,
                        'consistency_loss': args.consistency_loss,
                        'consistency_rampup': args.consistency_rampup,
                        'git_commit_hash': args.git_commit_hash,
                    }, fp, indent=2)

        use_base_dist = args.elbo_loss

        n_channels = 3
        flow_model = get_model(flow_args,
                               data_shape=(
                                   n_channels, args.img_size, args.img_size),
                               base_dist=use_base_dist)
        if not (args.resume or args.no_restore_flow_ckpt):
            path_check = '{}/check/checkpoint.pt'.format(args.flow_ckpt)
            checkpoint = torch.load(path_check, map_location=f'cuda:{rank}')
            if not use_base_dist:
                for k in tuple(checkpoint['model']):
                    if k.startswith('base_dist.'):
                        del checkpoint['model'][k]
            flow_model.load_state_dict(checkpoint['model'])
        flow_model.to(rank)

        n_classes = {
            'cub200': 200,
            'cifar10': 10,
            'mnist': 10,
            'cifar100': 100,
            'flowers': 102,
            'pets': 37,
            'imagenet': 1000,
        }[args.dataset]

        model = ProtoFlowGMM(
            model=flow_model,
            n_classes=n_classes,
            features_shape=flow_model.out_shape,
            protos_per_class=args.protos_per_class,
            gaussian_approach=args.gaussian_approach,
            likelihood_approach=args.likelihood_approach,
            proto_dropout_prob=args.proto_dropout_prob,
            z_dropout_prob=args.z_dropout_prob,
        )
        model.to(rank)

        if not args.debug:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

            model = DDP(model, device_ids=[rank],
                        find_unused_parameters=(args.trainable == 'gmm'),
                        broadcast_buffers=False)


        else:
            model.module = model

        assert args.batch_size % args.batch_steps == 0, 'batch step must divide batch size evenly'

        ds_train = get_dataset(
            args.dataset, train=True, use_shm=args.use_shm,
            transform=get_transform(args.interpolation, args.img_size,
                                    train=True,
                                    augmentation=args.augmentation,
                                    transform_twice=args.consistency_loss),
        )
        val_len = round(len(ds_train) * args.val_size)
        if val_len == 0 and args.val_size != 0 and len(ds_train) > 1:
            val_len = 1
        if val_len != 0:
            train_len = len(ds_train) - val_len
            g = torch.Generator().manual_seed(42)
            ds_train, ds_val = random_split(ds_train, [train_len, val_len],
                                            generator=g)
        else:
            ds_val = None

        if args.init_gmm:
            if args.resume:
                print(
                    f'Ignoring option --init_gmm {args.init_gmm} because we are '
                    f'resuming training')
            else:
                print('Load train data w/o augmentations')
                ds_train_no_aug = get_dataset(
                    args.dataset, train=True, use_shm=args.use_shm,
                    transform=get_transform(args.interpolation, args.img_size,
                                            train=False),
                )
                ds_train_no_aug = Subset(ds_train_no_aug, ds_train.indices)
                dl_train_no_aug = make_dataloader(
                    ds_train_no_aug, rank, world_size, args.batch_size,
                    drop_last=True, shuffle=False, persistent_workers=False,
                    num_workers=args.num_workers,
                    prefetch_cuda=args.prefetch_cuda,
                )
                init_gmms(rank, world_size, model, dl_train_no_aug, args,
                          method=args.init_gmm)

                del dl_train_no_aug, ds_train_no_aug

        dl_train = make_dataloader(ds_train, rank, world_size, args.batch_size,
                                   drop_last=True, shuffle=True,
                                   num_workers=args.num_workers,
                                   prefetch_cuda=args.prefetch_cuda)
        if ds_val is None:
            dl_val = None
        else:
            dl_val = make_dataloader(ds_val, rank, world_size, args.batch_size,
                                     drop_last=False, shuffle=False,
                                     num_workers=args.num_workers,
                                     prefetch_cuda=args.prefetch_cuda)

        with torch.no_grad() if args.gmm_em else nullcontext():
            train_loop(
                rank=rank,
                world_size=world_size,
                model=model,
                run_folder=run_folder,
                dl_train=dl_train,
                dl_val=dl_val,
                n_classes=n_classes,
                args=args,
            )

        if rank == 0:
            print('Test accuracy time')
        ds_test = get_dataset(
            args.dataset, train=False, use_shm=args.use_shm,
            transform=get_transform(args.interpolation, args.img_size,
                                    train=False),
        )
        dl_test = make_dataloader(ds_test, rank, world_size, args.batch_size,
                                  num_workers=args.num_workers,
                                  prefetch_cuda=args.prefetch_cuda)
        test_scores = test(
            rank=rank,
            model=model,
            dl=dl_test,
            n_classes=n_classes,
            args=args,
        )
        if rank == 0:
            print('EVALUATION STATS:')
            for name, score in test_scores.items():
                print(f'  {name}: {score.item():.3f}')
    finally:
        cleanup(args.debug)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train a ProtoFlow model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['pets', 'flowers', 'stl10', 'mnist', 'cifar10',
                                 'food', 'caltech101', 'imagenet', 'cifar100',
                                 'objectnet', 'aircraft', 'cub200'],
                        help='Dataset to use')

    parser.add_argument('--img_size', type=int, required=True,
                        help='Image size')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--batch_steps', '-s', type=int, default=1)
    parser.add_argument('--num_epochs', '-e', type=int, default=50)
    parser.add_argument('--warmup_epochs', '-w', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gmm_lr', type=float, default=1e-3)
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Resize interpolation type')
    parser.add_argument('--extra', type=str, default=None,
                        help='To append to the run folder name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume checkpoint')
    parser.add_argument('--no_restore_optimizer', action='store_true',
                        help='Do not restore the optimizer and scheduler')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA (exponential moving average) of model weights')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Optimizer weight decay')
    parser.add_argument('--trainable', default='gmm',
                        choices=('gmm', 'flow', 'all', 'all_means'))
    parser.add_argument('--save_k_best', default=2, type=int)
    parser.add_argument('--val_size', default=0.1, type=float)
    parser.add_argument('--clip_grad_norm', default=100.0, type=float)
    parser.add_argument('--augmentation', default='v1',
                        choices=['simple', 'v1', 'v2', 'timm-original',
                                 'timm-rand', 'timm-augmix'],
                        help='Augmentation type')
    parser.add_argument('--debug', action='store_true',
                        help='Debugging')
    parser.add_argument('--detect_autograd_anomaly', action='store_true',
                        help='Set autograd detect anomaly to true')

    parser.add_argument('--flow_ckpt', type=str, required=True)
    parser.add_argument('--no_restore_flow_ckpt', action='store_true')
    parser.add_argument('--gmm_em', action='store_true',
                        help='Train GMMs using EM only')
    parser.add_argument('--protos_per_class', type=int, default=10,
                        help='Number of prototypical distributions per class')
    parser.add_argument('--init_gmm', default=None,
                        choices=('kmeans',),
                        help='Init GMM means using this method')
    parser.add_argument('--gaussian_approach', default='GaussianMixture',
                        choices=('GaussianMixture', 'SSLGaussMixture',
                                 'GaussianMixtureConv2d'),
                        help='GMM approach')
    parser.add_argument('--likelihood_approach', default='total',
                        choices=('total', 'max'),
                        help='GMM likelihood approach per class')
    parser.add_argument('--proto_dropout_prob', default=None, type=float,
                        help='GMM prototype dropout prob')
    parser.add_argument('--z_dropout_prob', default=None, type=float,
                        help='Embedding dropout prob')
    parser.add_argument('--mu_loss', action='store_true',
                        help='Add cluster loss term for means of GMMs')
    parser.add_argument('--proto_loss', action='store_true',
                        help='Add elbo loss term for means of GMMs')
    parser.add_argument('--elbo_loss', action='store_true',
                        help='Add elbo loss term for z...')
    parser.add_argument('--elbo_loss2', action='store_true',
                        help='Add different elbo loss term for z...')
    parser.add_argument('--consistency_loss', action='store_true',
                        help='Add consistency loss term')
    parser.add_argument('--consistency_rampup', type=int, default=50,
                        help='Consistency loss rampup epochs')
    parser.add_argument('--diversity_loss', action='store_true',
                        help='Add intra-class prototype diversity loss term')

    parser.add_argument('--num_workers', default=None, type=int,
                        help='Number of workers per DataLoader')
    parser.add_argument('--prefetch_cuda', action='store_true',
                        help='Prefetch data to CUDA devices')
    parser.add_argument('--use_shm', action='store_true',
                        help='Used shared memory to move data before training for speed')

    args = parser.parse_args()
    print(args)

    run(args)


if __name__ == '__main__':
    main()
