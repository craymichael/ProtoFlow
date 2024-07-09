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

import math
import torch
from torch.nn import functional as F
from torchmetrics.metric import Metric


def elbo_bpd(log_prob: torch.Tensor, x_shape: torch.Size):
    return -log_prob.sum() / (math.log(2) * x_shape.numel())


class ElboBPD(Metric):
    is_differentiable = True
    full_state_update = False

    sum_lob_prob: torch.Tensor
    total: torch.Tensor

    def __init__(
            self,

            **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state('sum_lob_prob', default=torch.tensor(0.),
                       dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, log_prob: torch.Tensor, x_shape: torch.Size) -> None:
        self.sum_lob_prob += log_prob.sum()
        self.total += x_shape.numel()

    def compute(self) -> torch.Tensor:
        return -self.sum_lob_prob / (math.log(2) * self.total)


class CrossEntropy(Metric):
    is_differentiable = True
    full_state_update = False

    sum_xent: torch.Tensor
    total: torch.Tensor

    def __init__(
            self,
            weight=None,
            size_average=None,
            ignore_index=-100,
            label_smoothing=0.0,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        self.add_state('sum_xent', default=torch.tensor(0.),
                       dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, input: torch.Tensor, target: torch.Tensor) -> None:
        self.sum_xent += F.cross_entropy(
            input=input,
            target=target,
            weight=self.weight,
            size_average=self.size_average,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction='sum',
        )
        self.total += len(input)

    def compute(self) -> torch.Tensor:
        return self.sum_xent / self.total
