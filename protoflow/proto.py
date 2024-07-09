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

from typing import Union
from typing import Tuple
from typing import Optional
from typing import Sequence
from contextlib import nullcontext

import math

import torch
from torch import nn

from experiments.image.model.dense_flow import DenseFlow
from protoflow.gmm import GaussianMixture
from protoflow.gmm import SSLGaussMixture
from protoflow.gmm import GaussianMixtureConv2d


class ProtoFlowGMM(nn.Module):
    def __init__(
            self,
            model: DenseFlow,
            n_classes: int,
            features_shape: Sequence[int],
            protos_per_class: int = 10,
            likelihood_approach: str = 'total',
            gaussian_approach: str = 'GaussianMixture',
            proto_dropout_prob: Optional[float] = None,
            z_dropout_prob: Optional[float] = None,
    ):
        super().__init__()
        self.model = model

        assert likelihood_approach in {'total', 'max'}
        self.likelihood_approach = likelihood_approach
        self.z_dropout_prob = z_dropout_prob
        self.z_dropout = nn.Dropout(
            p=z_dropout_prob) if z_dropout_prob else nn.Identity()

        gmms = []
        for _ in range(n_classes):
            if gaussian_approach == 'GaussianMixture':
                gmm = GaussianMixture(
                    n_components=protos_per_class,
                    n_features=math.prod(features_shape),
                    requires_grad=True,
                    covariance_type='diag',
                    init_params='kmeans',
                    dropout_prob=proto_dropout_prob,
                )
            elif gaussian_approach == 'SSLGaussMixture':
                gmm = SSLGaussMixture(
                    means=torch.randn(protos_per_class,
                                      math.prod(features_shape)),
                    inv_cov_stds=None,
                    device=None,
                    dropout_prob=proto_dropout_prob,
                )
            elif gaussian_approach == 'GaussianMixtureConv2d':
                gmm = GaussianMixtureConv2d(
                    features_shape=features_shape,
                    n_components=protos_per_class,
                    dropout_prob=proto_dropout_prob,
                )
            else:
                raise NotImplementedError(gaussian_approach)
            gmms.append(gmm)
        self.gmms = nn.ModuleList(gmms)

    def forward(self,
                x: torch.Tensor,
                flow_grad=False,
                ret_log_prob=False,
                ret_z=False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

        if not len(x):
            return torch.empty((0, len(self.gmms)), dtype=x.dtype,
                               device=x.device)
        with nullcontext() if flow_grad else torch.no_grad():
            z, log_prob = self.model.log_prob(x, return_z=True)
        z_flat = z.flatten(1)
        z_flat = self.z_dropout(z_flat)
        if self.likelihood_approach == 'total':
            preds = [
                gmm.score_samples(z_flat)
                for gmm in self.gmms
            ]
        elif self.likelihood_approach == 'max':
            preds = [

                gmm.log_prob_all(z_flat).max(dim=1).values
                for gmm in self.gmms
            ]
        else:
            raise NotImplementedError
        preds = torch.stack(preds, dim=1)
        if ret_log_prob:
            if ret_z:
                return preds, log_prob, z
            return preds, log_prob
        else:
            if ret_z:
                return preds, z
            return preds
