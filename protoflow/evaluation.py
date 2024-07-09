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
from tqdm.auto import tqdm

import torch
import torchmetrics

from protoflow.metrics import ElboBPD
from protoflow.metrics import CrossEntropy


def test(rank, model, dl, n_classes, args, num_samples=1, ten_crop=False,
         calibration_metrics=False, multi_transform_k=None, log_suffix=None):
    was_training = model.training

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
        'bpd': ElboBPD().to(rank),
        'nll': CrossEntropy().to(rank),
    }
    if calibration_metrics:
        metrics.update({
            'ece': torchmetrics.CalibrationError(
                task='multiclass',
                n_bins=15,
                norm='l1',
                num_classes=n_classes,
            ).to(rank),
            'mce': torchmetrics.CalibrationError(
                task='multiclass',
                n_bins=15,
                norm='max',
                num_classes=n_classes,
            ).to(rank),
        })
    model.eval()

    inner_batch_size = args.batch_size // args.batch_steps
    with torch.no_grad():
        desc = 'Evaluating' + (f' {log_suffix}' if log_suffix else '') + '...'
        for batch_idx, (images, labels) in enumerate(tqdm(
                dl, disable=rank != 0, desc=desc)):
            if isinstance(images, (tuple, list)):
                assert len(images) == 2
                images = images[0]
            images = images.to(rank)
            labels = labels.to(rank)

            for i in range(args.batch_steps):
                idxs_step = slice(
                    i * inner_batch_size, (i + 1) * inner_batch_size)
                images_step = images[idxs_step]
                if not len(images_step):
                    break
                labels_step = labels[idxs_step]
                preds_agg = None
                log_prob_agg = None
                x_shape = None
                for _ in range(num_samples):
                    n_augs = images_step.shape[1] if multi_transform_k else 1
                    for aug_idx in range(n_augs):
                        if multi_transform_k:
                            images_step_aug = images_step[:, aug_idx]
                        else:
                            images_step_aug = images_step
                        n_crops = images_step_aug.shape[1] if ten_crop else 1
                        divisor = num_samples * n_crops * n_augs
                        for crop_idx in range(n_crops):
                            if ten_crop:
                                images_step_crop = images_step_aug[:, crop_idx]
                            else:
                                images_step_crop = images_step_aug
                            preds, log_prob = model(
                                images_step_crop, flow_grad=False,
                                ret_log_prob=True, ret_z=False,
                            )

                            if preds_agg is None:
                                preds_agg = preds / divisor
                                log_prob_agg = log_prob / divisor
                                x_shape = images_step_crop.shape
                            else:
                                preds_agg += preds / divisor
                                log_prob_agg += log_prob / divisor

                for m_name, metric in metrics.items():
                    if m_name == 'bpd':

                        metric(
                            log_prob_agg + preds_agg.max(dim=1).values,
                            x_shape
                        )
                    else:
                        metric(preds_agg, labels_step)

        scores = {}
        for name, metric in metrics.items():
            score = metric.compute()
            if name.startswith('acc'):
                score *= 100
            scores[name] = score

    if was_training:
        model.train()

    return scores
