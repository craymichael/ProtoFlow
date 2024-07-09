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
from torch import nn


class LogDropout(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError(
                f'dropout probability has to be between 0 and 1, but got {p}'
            )
        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training:
            if not self.inplace:
                input = torch.clone(input)

            replace_value = torch.finfo(input.dtype).min
            mask = torch.rand_like(input) < self.p
            input -= math.log(1 - self.p)
            input[mask] = replace_value
        return input

    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'
