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

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchtnt.utils.data import CudaDataPrefetcher

from protoflow.utils import num_cpus


class CudaDataPrefetcherWrapped(CudaDataPrefetcher):

    def __len__(self):
        return len(self.data_iterable)

    @property
    def sampler(self):
        return self.data_iterable.sampler

    @property
    def batch_size(self):
        return self.data_iterable.batch_size


def make_dataloader(ds, rank, world_size, batch_size, drop_last=False,
                    shuffle=False, persistent_workers=True, num_workers=None,
                    prefetch_cuda=False):
    if num_workers is None:
        num_workers = max((num_cpus() - 2) // (2 * world_size), 2)
    dl_kws = dict(batch_size=batch_size,
                  num_workers=num_workers,
                  pin_memory=False,
                  prefetch_factor=2,
                  persistent_workers=persistent_workers,
                  drop_last=drop_last,
                  shuffle=False)
    sampler_kws = dict(num_replicas=world_size, rank=rank,
                       drop_last=drop_last, shuffle=shuffle)
    dl = DataLoader(ds,
                    sampler=DistributedSampler(ds, **sampler_kws),
                    **dl_kws)
    if prefetch_cuda:
        device = torch.device(f'cuda:{rank}')
        return CudaDataPrefetcherWrapped(dl, device=device, num_prefetch_batches=2)
    else:
        return dl
