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
from typing import Dict

import os
from argparse import Namespace
from functools import wraps

__all__ = ('profile', 'num_cpus', 'dict_to_namespace', 'convert_legacy_config')

PROFILE = bool(os.getenv('DO_PROFILE', False))

if PROFILE:
    print('Profiling!')
    from line_profiler_pycharm import profile
else:
    def profile(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


def num_cpus() -> int:
    if 'SLURM_CPUS_PER_TASK' in os.environ:

        try:
            return int(os.environ['SLURM_CPUS_PER_TASK'])
        except ValueError:
            raise RuntimeError(f'Detected SLURM environment but '
                               f'$SLURM_CPUS_PER_TASK is not an int: '
                               f'{os.environ["SLURM_CPUS_PER_TASK"]}')
    elif 'JOB_ID' in os.environ:

        base_err = ('Inferred that you are in an SGE environment (because '
                    f'$JOB_ID is set as {os.environ["JOB_ID"]}) but $NSLOTS '
                    f'is not ')
        try:
            return int(os.environ['NSLOTS'])
        except KeyError:
            raise RuntimeError(base_err + 'set!')
        except ValueError:
            raise RuntimeError(base_err + f'an int ({os.environ["NSLOTS"]})!')
    else:

        return os.cpu_count()


def dict_to_namespace(d: Dict) -> Namespace:
    n = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            v = dict_to_namespace(v)
        setattr(n, k, v)
    return n


def convert_legacy_config(config: Dict) -> Dict:
    config_ = config.copy()
    if 'simple_aug' in config:
        simple_aug = config_.pop('simple_aug')
        config_['augmentation'] = 'simple' if simple_aug else 'v1'
    return config_
