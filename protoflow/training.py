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

from typing import Optional

import math
import os
from os import path as osp

from functools import partial

from tqdm import tqdm

from diffusers import get_cosine_schedule_with_warmup

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import get_ema_multi_avg_fn
from torch.optim.swa_utils import update_bn

import torchmetrics

from protoflow.kmeans import kmeans
from protoflow.metrics import elbo_bpd
from protoflow.utils import PROFILE, profile
from protoflow.evaluation import test
from protoflow.gmm import GaussianMixture


def _convert_image_to_rgb(image):
    return image.convert('RGB')


class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class MultipleTransform:
    def __init__(self, transform, k):
        assert k >= 1
        self.transform = transform
        self.k = k

    def __call__(self, x):
        return torch.stack([
            self.transform(x)
            for _ in range(self.k)
        ], dim=0)


def linear_rampup(final_value, epoch, num_epochs, start_epoch=0):
    t = (epoch - start_epoch + 1) / num_epochs
    if t > 1:
        t = 1.
    return t * final_value


def z_log_prob(all_log_probs,
               y: Optional[torch.Tensor] = None):
    if y is not None:
        log_probs = torch.empty(len(all_log_probs),
                                device=all_log_probs.device,
                                dtype=all_log_probs.dtype)
        mask = torch.where(y == -1)[0]
        if len(mask):
            log_probs[mask] = torch.logsumexp(all_log_probs[mask, :], dim=1)
        for c in range(all_log_probs.shape[1]):
            mask = torch.where(y == c)[0]
            log_probs[mask] = all_log_probs[mask, c]
        return log_probs
    else:
        mixture_log_probs = torch.logsumexp(all_log_probs, dim=1)
        return mixture_log_probs


def consistency_loss(all_log_probs, z, sldj, x_shape=None, y=None, k=256):
    ORIGINAL = False
    if ORIGINAL:
        z = z.reshape((z.shape[0], -1))
        prior_ll = z_log_prob(all_log_probs, y)
        corrected_prior_ll = prior_ll - math.log(k) * math.prod(z.size()[1:])

        ll = corrected_prior_ll + sldj
        nll = -ll.mean()
    else:
        prior_ll = z_log_prob(all_log_probs,
                              y)
        ll = prior_ll + sldj
        nll = -ll.sum() / (math.log(2) * x_shape.numel())

    return nll


def var_determinant(var, covariance_type):
    if covariance_type == 'full':
        det = torch.linalg.det(var)
    elif covariance_type == 'diag':
        det = torch.sum(torch.log(var), dim=-1).exp()
    else:
        raise NotImplementedError

    if det.ndim == 2:
        det = det.squeeze(dim=0)
    return det


def var_inverse(var, covariance_type):
    if covariance_type == 'full':
        inv = torch.linalg.inv(var)

        if inv.ndim == 4:
            inv = inv.squeeze(dim=0)
        return inv
    elif covariance_type == 'diag':
        inv = (1 / var)

        if inv.ndim == 3:
            inv = inv.squeeze(dim=0)
        return inv
    else:
        raise NotImplementedError


def hellinger_distance(
        mu1, var1, var_det1,
        mu2, var2, var_det2,
        covariance_type,
):
    d = mu1.shape[-1]

    var_d2 = (var1 + var2) / 2
    mu_d = mu1 - mu2
    inv_var_d2 = var_inverse(var_d2, covariance_type)
    if covariance_type == 'full':
        exp_term = torch.exp((-1 / (8 * d)) * (mu_d @ inv_var_d2) @ mu_d)
    elif covariance_type == 'diag':
        exp_term = torch.exp((-1 / (8 * d)) * (mu_d * inv_var_d2) @ mu_d)
    else:
        raise NotImplementedError
    det_term = (var_det1 ** (1 / 4) * var_det2 ** (1 / 4) /
                var_determinant(var_d2, covariance_type) ** (1 / 2)) ** (1 / d)
    return (
            1 - det_term *
            exp_term
    )


def diversity_loss(gmms):
    loss = 0
    for gmm in gmms:
        if isinstance(gmm, GaussianMixture):
            var, mu = gmm.var.squeeze(0), gmm.mu.squeeze(0)
        else:
            var, mu = gmm.var, gmm.mu
        var_det = var_determinant(var, gmm.covariance_type)

        n_combs = gmm.n_components * (gmm.n_components - 1) // 2
        for i in range(gmm.n_components - 1):
            for j in range(i + 1, gmm.n_components):
                distance = hellinger_distance(
                    mu[i], var[i], var_det[i],
                    mu[j], var[j], var_det[j],
                    covariance_type=gmm.covariance_type,
                )

                loss += -distance / n_combs
    return loss / len(gmms)


def identity(x):
    return x


def get_transform(interpolation='bicubic', size=64, train=True,
                  augmentation='v1', transform_twice=False, ten_crop=False,
                  multi_transform_k=None):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    from denseflow.data.transforms import Quantize

    INTERPOLATIONS = {
        'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'lanczos': InterpolationMode.LANCZOS,
    }

    interpolation_str = interpolation
    interpolation = INTERPOLATIONS[interpolation]

    assert not (multi_transform_k is not None and transform_twice), (
        'Cannot specify both multi_transform_k and transform_twice')
    if transform_twice:
        wrap = TransformTwice
    elif multi_transform_k:
        wrap = partial(MultipleTransform, k=multi_transform_k)
    else:
        wrap = identity

    if ten_crop:
        crop = T.Compose([
            T.TenCrop(size),
            T.Lambda(torch.stack),
        ])
    else:
        crop = T.CenterCrop(size)
    if train:
        if augmentation == 'simple':
            return wrap(T.Compose([
                T.Resize(size, interpolation=interpolation),
                T.RandomHorizontalFlip(),
                T.Pad(int(math.ceil(size * 0.04)),
                      padding_mode='edge'),
                T.RandomAffine(degrees=0,
                               translate=(0.04, 0.04)),
                _convert_image_to_rgb,
                T.ToTensor(),
                Quantize(num_bits=8),
                crop,
            ]))
        elif augmentation == 'v1':
            if ten_crop:
                crop = T.Compose([
                    T.CenterCrop(round(size * 1.2)),
                    T.TenCrop(size),
                    T.Lambda(torch.stack),
                ])
            return wrap(T.Compose([
                T.Resize(size, interpolation=interpolation),
                T.RandomHorizontalFlip(0.5),

                T.Pad(int(math.ceil(size / 2)),

                      padding_mode='edge'),
                T.RandomAffine(degrees=15,
                               translate=(0.04, 0.04),
                               shear=10),

                _convert_image_to_rgb,
                T.ToTensor(),
                Quantize(num_bits=8),
                crop,
            ]))
        elif augmentation == 'v2':
            from timm.data.transforms import RandomResizedCropAndInterpolation
            if ten_crop:
                crop = T.Compose([
                    T.CenterCrop(round(size * 1.2)),
                    T.TenCrop(size),
                    T.Lambda(torch.stack),
                ])
            return wrap(T.Compose([

                RandomResizedCropAndInterpolation(
                    size, interpolation=interpolation_str),
                T.RandomHorizontalFlip(0.5),

                T.Pad(int(math.ceil(size / 2)),

                      padding_mode='edge'),
                T.RandomAffine(degrees=15,
                               translate=(0.04, 0.04),
                               shear=10),
                T.RandomPerspective(p=1 / 3,
                                    distortion_scale=0.2),
                _convert_image_to_rgb,
                T.ToTensor(),
                Quantize(num_bits=8),
                crop,
            ]))
        elif augmentation.startswith('timm-'):
            from timm.data.auto_augment import rand_augment_transform, \
                augment_and_mix_transform, auto_augment_transform
            from timm.data.transforms import RandomResizedCropAndInterpolation, \
                str_to_pil_interp
            from timm.data.random_erasing import RandomErasing

            if ten_crop:
                raise NotImplementedError

            scale = (0.08, 1.0)
            ratio = (3. / 4., 4. / 3.)
            color_jitter = 0.4
            primary_tfl = [
                RandomResizedCropAndInterpolation(
                    size,
                    scale=scale,
                    ratio=ratio,
                    interpolation=interpolation_str,
                ),
                T.RandomHorizontalFlip(p=0.5),
            ]

            secondary_tfl = []
            disable_color_jitter = False
            auto_augment = {
                'timm-rand': 'rand-m9-mstd0.5-inc1',
                'timm-augmix': 'augmix-m9-mstd0.5',
                'timm-original': 'original',
            }[augmentation]
            if auto_augment:

                disable_color_jitter = '3a' not in auto_augment
                if isinstance(size, (tuple, list)):
                    size_min = min(size)
                else:
                    size_min = size
                aa_params = dict(
                    translate_const=int(size_min * 0.45),

                )
                if interpolation_str and interpolation_str != 'random':
                    aa_params['interpolation'] = str_to_pil_interp(
                        interpolation_str)
                if auto_augment.startswith('rand'):
                    secondary_tfl += [
                        rand_augment_transform(auto_augment, aa_params)]
                elif auto_augment.startswith('augmix'):
                    aa_params['translate_pct'] = 0.3
                    secondary_tfl += [
                        augment_and_mix_transform(auto_augment, aa_params)]
                else:
                    secondary_tfl += [
                        auto_augment_transform(auto_augment, aa_params)]

            if color_jitter is not None and not disable_color_jitter:

                if isinstance(color_jitter, (list, tuple)):

                    assert len(color_jitter) in (3, 4)
                else:

                    color_jitter = (float(color_jitter),) * 3
                secondary_tfl += [T.ColorJitter(*color_jitter)]

            final_tfl = [
                _convert_image_to_rgb,
                T.ToTensor(),
            ]

            re_prob = 0.
            re_mode = 'pixel'
            re_count = 1
            re_num_splits = 0
            if re_prob > 0.:
                final_tfl += [
                    RandomErasing(
                        re_prob,
                        mode=re_mode,
                        max_count=re_count,
                        num_splits=re_num_splits,
                        device='cpu',
                    )
                ]
            final_tfl += [Quantize(num_bits=8)]
            return wrap(
                T.Compose(primary_tfl + secondary_tfl + final_tfl))
        else:
            raise NotImplementedError(augmentation)
    else:
        return wrap(T.Compose([
            T.Resize(size, interpolation=interpolation),
            _convert_image_to_rgb,
            T.ToTensor(),
            Quantize(num_bits=8),
            crop,
        ]))


def all_gather(tensor: torch.Tensor, world_size: int, disable: bool = False):
    if disable:
        gather_t = [tensor]
    else:
        gather_t = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_t, tensor)
    return torch.stack(gather_t, dim=0)


@profile
def train_loop(rank, world_size, model, run_folder, dl_train, dl_val,
               n_classes, args):
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=osp.join(run_folder, 'tb_logs'))

    inner_batch_size = args.batch_size // args.batch_steps

    global_step = 0
    start_epoch = 0
    checkpoints = {}
    parameters = []
    if args.trainable in {'all', 'gmm'}:
        parameters.append({
            'params': model.module.gmms.parameters(),
            'lr': args.gmm_lr,
            'weight_decay': 0.,
        })
    elif args.trainable == 'all_means':
        parameters.append({
            'params': (gmm.mu for gmm in model.module.gmms),
            'lr': args.gmm_lr,
            'weight_decay': 0.,
        })

    if args.trainable in {'all', 'all_means', 'flow'}:
        model.module.model.train()
        parameters.append({'params': model.module.model.parameters()})
        flow_grad = True
    else:
        model.module.model.eval()
        flow_grad = False

    optimizer = torch.optim.AdamW(parameters, lr=args.lr,
                                  weight_decay=args.weight_decay)
    num_warmup_steps = round(args.warmup_epochs * len(dl_train))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(dl_train) * args.num_epochs),
    )

    checkpoint = None
    if args.resume:
        if rank == 0:
            print(f'Resuming from checkpoint {args.resume}')
        checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
        global_step = checkpoint['global_step']
        start_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['model'])
        if not args.no_restore_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.use_ema:
        ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(0.999),
        )
        if args.resume and 'model_ema' in checkpoint:
            ema_model.module.module.load_state_dict(
                checkpoint['model_ema'])

    all_labels = all_z = None
    if args.gmm_em:
        if rank > 0:
            raise NotImplementedError('Multi-GPU EM')
        args.num_epochs = start_epoch + 1
        all_labels = []
        all_z = []

    metrics = {
        'acc': torchmetrics.Accuracy(
            task='multiclass',
            num_classes=n_classes,
        ).to(rank),
        'acc5': torchmetrics.Accuracy(
            task='multiclass',
            num_classes=n_classes,
            top_k=5,
        ).to(rank),
    }

    loss_names = ['loss_xent']
    if args.elbo_loss:
        loss_names.append('loss_elbo')
    if args.elbo_loss2:
        loss_names.append('loss_elbo2')
    if args.mu_loss:
        loss_names.append('loss_cluster')
    if args.proto_loss:
        loss_names.append('loss_proto')
    if args.consistency_loss:
        loss_names.append('loss_consistency')
    if args.diversity_loss:
        loss_names.append('loss_diversity')
    loss_metrics = {
        loss_name: torchmetrics.SumMetric().to(rank)
        for loss_name in loss_names
    }

    logs = {}
    nan_loss_patience = 3
    nan_loss_count = 0
    for epoch in range(start_epoch, args.num_epochs):
        dl_train.sampler.set_epoch(epoch)

        consistency_weight = linear_rampup(
            1.0, epoch, args.consistency_rampup, start_epoch)

        pbar = tqdm(total=len(dl_train), disable=rank != 0,
                    desc=f'Epoch {epoch + 1}/{args.num_epochs}')
        pbar.set_postfix(**logs)
        avg_loss = batch_idx = 0
        for batch_idx, (images, labels) in enumerate(dl_train):
            if args.gmm_em:
                all_labels.append(labels)
            if args.consistency_loss:
                images, images2 = images
                images2 = images2.to(rank)
            images = images.to(rank)
            labels = labels.to(rank)

            for i in range(args.batch_steps):
                idxs_step = slice(
                    i * inner_batch_size, (i + 1) * inner_batch_size)
                images_step = images[idxs_step]
                labels_step = labels[idxs_step]
                preds, log_prob, z = model(images_step, flow_grad=flow_grad,
                                           ret_log_prob=True, ret_z=True)

                losses = {'loss_xent': (F.cross_entropy(preds, labels_step) /
                                        args.batch_steps)}
                if args.elbo_loss:
                    losses['loss_elbo'] = (
                            elbo_bpd(log_prob, images_step.shape) /
                            args.batch_steps)
                if args.consistency_loss:
                    model.eval()

                    with torch.no_grad():
                        images2_step = images2[idxs_step]
                        preds2 = model(images2_step).argmax(dim=1)
                    model.train()

                    losses['loss_consistency'] = consistency_loss(
                        all_log_probs=preds, z=z, sldj=log_prob,
                        x_shape=images_step.shape, y=preds2
                    ) / args.batch_steps * consistency_weight
                if args.elbo_loss2:
                    losses['loss_elbo2'] = elbo_bpd(
                        log_prob + preds[torch.arange(len(preds)), labels_step],
                        images_step.shape
                    ) / args.batch_steps
                if args.diversity_loss:
                    losses['loss_diversity'] = (
                            diversity_loss(model.module.gmms) /
                            args.batch_steps
                    )
                if args.mu_loss:

                    classes_covered = torch.unique(labels_step)
                    loss_cluster = None
                    for c in classes_covered:
                        z_c = z[labels_step == c].flatten(1)
                        mu_c = model.module.gmms[c].mu.squeeze(0)
                        dists = torch.cdist(z_c, mu_c)

                        min_dists = dists.min(dim=0).values
                        if loss_cluster is None:
                            loss_cluster = min_dists.mean()
                        else:
                            loss_cluster += min_dists.mean()

                    losses['loss_cluster'] = loss_cluster / (
                            len(classes_covered) * args.batch_steps)
                if args.proto_loss:
                    loss_proto = None
                    img_size = images.shape[-1]
                    for gmm in model.module.gmms:
                        mu = gmm.mu.reshape(-1, *model.module.model.out_shape)

                        loss_proto_k = elbo_bpd(
                            model.module.model.base_dist.log_prob(mu),
                            torch.Size([len(mu), 3, img_size, img_size]),
                        )
                        if loss_proto is None:
                            loss_proto = loss_proto_k
                        else:
                            loss_proto += loss_proto_k

                    losses['loss_proto'] = loss_proto / (
                            len(model.module.gmms) * args.batch_steps)
                loss = sum(losses.values())

                if torch.isnan(loss):
                    nan_loss_count += 1
                    nan_losses = []
                    for loss_name, loss_value in losses.items():
                        if torch.isnan(loss_value):
                            nan_losses.append(loss_name)

                    print(f'The following losses were NaN: {nan_losses}')
                    if nan_loss_count >= nan_loss_patience:
                        raise RuntimeError('NaN loss!')
                    else:
                        print(f'Encountered {nan_loss_count} batches in a row '
                              f'of NaN loss!')
                else:
                    nan_loss_count = 0

                for loss_name, loss_metric in loss_metrics.items():
                    loss_metric(losses[loss_name].detach() / world_size)

                if args.gmm_em:
                    all_z.append(z.detach().cpu())
                else:
                    loss.backward()

                for metric in metrics.values():
                    metric(preds, labels_step)

            last_lr = lr_scheduler.get_last_lr()[0]

            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip_grad_norm)

            batch_loss = 0
            for loss_name, loss_metric in loss_metrics.items():
                loss_value_agg = loss_metric.compute()
                batch_loss += loss_value_agg
                if rank == 0 and not (PROFILE or args.gmm_em):
                    writer.add_scalar(f'{loss_name}/train',
                                      loss_value_agg,
                                      global_step)
                logs[loss_name] = loss_value_agg.item()
            avg_loss += batch_loss
            if rank == 0 and not (PROFILE or args.gmm_em):
                writer.add_scalar('loss/train', batch_loss, global_step)
                writer.add_scalar('lr', last_lr, global_step)

            pbar.update(1)
            logs.update(
                loss=float(batch_loss),
                avg_loss=float(avg_loss) / (batch_idx + 1),
                lr=last_lr,
            )
            for name, metric in metrics.items():
                score = metric.compute()
                if name.startswith('acc'):
                    score *= 100
                logs[name] = score.item()
                if rank == 0 and not (PROFILE or args.gmm_em):
                    writer.add_scalar(f'{name}/train', score, global_step)
            pbar.set_postfix(**logs)

            for loss_metric in loss_metrics.values():
                loss_metric.reset()

            for metric in metrics.values():
                metric.reset()

            global_step += 1

            if not args.gmm_em:
                optimizer.step()
                if args.use_ema:
                    ema_model.update_parameters(model)
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if args.gaussian_approach != 'GaussianMixtureConv2d':
                    for gmm in model.module.gmms:
                        clip_val = 2 * torch.finfo(gmm.var.dtype).tiny
                        gmm.var.data = gmm.var.clip(min=clip_val)

        if dl_val is not None:
            val_scores = test(
                rank=rank,
                model=model,
                dl=dl_val,
                n_classes=n_classes,
                args=args,
            )
            if args.use_ema:
                val_scores_ema = test(
                    rank=rank,
                    model=ema_model,
                    dl=dl_val,
                    n_classes=n_classes,
                    args=args,
                    log_suffix='EMA'
                )
            if rank == 0:
                for name, score in val_scores.items():
                    logs[f'val_{name}'] = score.item()
                    if not (PROFILE or args.gmm_em):
                        writer.add_scalar(f'{name}/val', score, global_step)
                if args.use_ema:
                    for name, score in val_scores_ema.items():
                        logs[f'val_ema_{name}'] = score.item()
                        if not (PROFILE or args.gmm_em):
                            writer.add_scalar(f'{name}/val_ema', score,
                                              global_step)
        if rank == 0 and not (PROFILE or args.gmm_em):
            avg_loss = float(avg_loss / (batch_idx + 1))
            save_path = osp.join(
                run_folder, f'checkpoint_{epoch:05d}_{avg_loss:.5g}.pt')
            write_checkpoint = False
            to_delete = None
            if len(checkpoints) < args.save_k_best:
                write_checkpoint = True
            else:
                worst_score_epoch = None
                for (score, epoch_prev), path in checkpoints.items():
                    if score > avg_loss and (
                            worst_score_epoch is None or
                            score > worst_score_epoch[0]
                    ):
                        worst_score_epoch = (score, epoch_prev)
                        to_delete = path
                if worst_score_epoch is not None:
                    write_checkpoint = True
                    del checkpoints[worst_score_epoch]

            if write_checkpoint:
                tqdm.write(f'Write checkpoint to {save_path}')
                checkpoints[(avg_loss, epoch)] = save_path
                state_dict = {
                    'global_step': global_step,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch + 1,
                }
                if args.use_ema:
                    state_dict.update(
                        model_ema=ema_model.module.module.state_dict(),
                        model_ema_updated_bn=False,
                    )
                torch.save(state_dict, save_path)
            if to_delete:
                tqdm.write(f'Delete checkpoint from {to_delete}')
                os.remove(to_delete)

    if args.gmm_em:
        if rank > 0:
            raise NotImplementedError('More than 1 GPU...')

        print('Fit GMMs using EM')
        model.cpu()

        all_z = torch.cat(all_z, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        for c, gmm in enumerate(model.module.gmms):
            z_c = all_z[all_labels == c]
            if not len(z_c):
                print(f'None of class {c}!')
                continue
            gmm.fit(z_c.flatten(1),
                    warm_start=True, delta=1e-3, n_iter=1000)
        model.to(rank)
    elif args.use_ema:
        if args.consistency_loss:

            print('WARNING: cannot update EMA batch norm statistics now '
                  'because the dataloader uses TransformTwice...')
        else:
            print('Updating EMA batch norm stats')

            update_bn(dl_train, ema_model, device=rank)

    if rank == 0 and not PROFILE:
        save_path = osp.join(run_folder, 'checkpoint_final.pt')
        tqdm.write(f'Write checkpoint to {save_path}')
        state_dict = {
            'global_step': global_step,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': args.num_epochs,
        }
        if args.use_ema and not args.gmm_em:
            state_dict.update(
                model_ema=ema_model.module.module.state_dict(),
                model_ema_updated_bn=True,
            )
        torch.save(state_dict, save_path)


def init_gmms(rank, world_size, model, dl, args, method='kmeans'):
    assert method in {'kmeans'}

    flow_model = model.module.model
    was_training = flow_model.training
    flow_model.eval()

    inner_batch_size = args.batch_size // args.batch_steps

    all_z = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(
                tqdm(dl, desc='Gathering embeddings', disable=rank != 0)):
            images = images.to(rank)
            labels = labels.to(rank)

            for i in range(args.batch_steps):
                idxs_step = slice(
                    i * inner_batch_size, (i + 1) * inner_batch_size)
                images_step = images[idxs_step]
                if not len(images_step):
                    break
                labels_step = labels[idxs_step]
                z, log_prob = flow_model.log_prob(images_step, return_z=True)

                all_z.append(
                    all_gather(z, world_size, args.debug).reshape(
                        -1, *z.shape[1:]).cpu()
                )
                all_labels.append(
                    all_gather(labels_step, world_size, args.debug).reshape(
                        -1, *labels_step.shape[1:]).cpu()
                )

    all_z = torch.cat(all_z, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    for c, gmm in enumerate(model.module.gmms):
        _, cluster_centers = kmeans(
            X=all_z[all_labels == c].flatten(1),
            num_clusters=gmm.n_components,
            distance='euclidean',
            device=all_z.device,
            tol=1e-4,
            tqdm_flag=(rank == 0),
            iter_limit=0,
            seed=69,
        )

        if args.gaussian_approach == 'GaussianMixtureConv2d':
            device = gmm.pi.device
            for i in range(gmm.n_components):
                center = cluster_centers[i]
                center = center.reshape(gmm.features_shape).mean(dim=(1, 2))
                gmm.gaussians[i].loc.data = center.reshape(
                    gmm.gaussians[i].loc.shape).to(device)
        else:
            gmm.mu.data = cluster_centers.reshape(gmm.mu.shape).to(
                gmm.mu.device)

    if was_training:
        flow_model.train()
