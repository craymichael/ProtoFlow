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
import json
import shutil
import pandas as pd
import torch
from torch.utils import data
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision import datasets
from torchvision.transforms.functional import get_dimensions

DATASET_ROOT = os.getenv('DATASET_ROOT', './data/')


class CUB200_2011(data.Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            loader=default_loader,
            download=True,
            yield_img_id=False,
            yield_orig_shape=False,
            crop_to_bbox=False,
            part_annotations=False,
    ):
        if target_transform is not None:
            raise NotImplementedError
        self.root = osp.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.train = train
        self.crop_to_bbox = crop_to_bbox
        self.part_annotations = part_annotations
        self.yield_img_id = yield_img_id
        self.yield_orig_shape = yield_orig_shape

        if download and self._download():
            pass
        elif not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. You can use download=True to download it')

    def _load_metadata(self):
        df_images = pd.read_csv(
            osp.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
            names=['img_id', 'filepath']
        )
        df_image_class_labels = pd.read_csv(
            osp.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'target']
        )
        df_train_test_split = pd.read_csv(
            osp.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ', names=['img_id', 'is_training_img']
        )

        self.data = df_images.merge(df_image_class_labels, on='img_id')
        self.data = self.data.merge(df_train_test_split, on='img_id')

        if self.crop_to_bbox:
            df_bbox = pd.read_csv(
                osp.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
                sep=' ', names=['img_id', 'x', 'y', 'w', 'h']
            )
            df_bbox_int = df_bbox.astype(int)
            assert (df_bbox == df_bbox_int).all().all()
            self.data = self.data.merge(df_bbox_int, on='img_id')

        if self.part_annotations:

            df_parts = pd.read_csv(
                osp.join(self.root, 'CUB_200_2011', 'parts', 'part_locs.txt'),
                sep=' ',
                names=['img_id', 'part_id', 'x', 'y', 'visible'],
            )
            df_parts_int = df_parts.astype(int)
            assert (df_parts.astype(float) == df_parts_int).all().all()
            df_parts = df_parts_int
            df_parts['part_id_orig'] = df_parts['part_id']
            df_parts['part_id'] -= 1
            df_parts['part_id'] = df_parts['part_id'].replace(
                {
                    0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 5,
                    6: 6,
                    7: 7,
                    8: 8,
                    9: 9,
                    10: 6,
                    11: 7,
                    12: 8,
                    13: 10,
                    14: 11,
                }
            )
            self.df_parts = df_parts
        else:
            self.df_parts = None

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = osp.join(self.root, self.base_folder, row.filepath)
            if not osp.isfile(filepath):
                print(f'{filepath} is not a file...')
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print(
                f'{type(self).__name__} Files already downloaded and ' f'verified')
            return True

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(osp.join(self.root, self.filename), 'r:gz') as tar:
            tar.extractall(path=self.root)
        return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = osp.join(self.root, self.base_folder, sample.filepath)

        target = sample.target - 1
        img = self.loader(path)

        if self.yield_orig_shape:
            orig_shape = torch.tensor(get_dimensions(img))

        if self.crop_to_bbox:
            img = img.crop(
                (sample.x, sample.y, sample.x + sample.w, sample.y + sample.h))

        if self.transform is not None:
            img = self.transform(img)

        if self.yield_img_id:
            if self.yield_orig_shape:
                return sample.img_id, orig_shape, img, target
            return sample.img_id, img, target
        elif self.yield_orig_shape:
            return orig_shape, img, target
        return img, target


_SHM_DATA_ROOT = osp.join('/dev/shm/', '.protoflow_data_shm')
_SHM_LOCK_PATH = '/tmp/.protoflow_dist_data_lock'
_SHM_META_PATH = '/tmp/.protoflow_dist_data_meta.json'
_SHM_ACTIVE_PATHS = []


def cleanup_shm_data():
    global _SHM_ACTIVE_PATHS

    if not _SHM_ACTIVE_PATHS:
        return

    from filelock import FileLock

    with FileLock(_SHM_LOCK_PATH):
        shm_meta = load_shm_meta()
        for active_path in _SHM_ACTIVE_PATHS:

            shm_meta[active_path] -= 1
            if shm_meta[active_path] > 0:
                continue

            print(f'>>> Removing data from shared memory '
                  f'({active_path})')
            shutil.rmtree(active_path)

        write_shm_meta(shm_meta)

        _SHM_ACTIVE_PATHS = []


def load_shm_meta():
    if not osp.exists(_SHM_META_PATH):
        return {}
    with open(_SHM_META_PATH, 'r') as fp:
        return json.load(fp)


def write_shm_meta(meta):
    with open(_SHM_META_PATH, 'w') as fp:
        return json.dump(meta, fp)


def get_dataset(name: str, train=False, transform=None,
                target_transform=None, use_shm=False):
    dataset_root_subdir = osp.join(DATASET_ROOT, name.lower())

    if use_shm:
        from filelock import FileLock

        with FileLock(_SHM_LOCK_PATH):

            new_dataset_root_subdir = osp.join(_SHM_DATA_ROOT, name.lower())

            shm_meta = load_shm_meta()
            shm_data_use_count = shm_meta.get(new_dataset_root_subdir, 0)
            if shm_data_use_count < 1:
                print(f'>>> Copying data to shared memory '
                      f'({dataset_root_subdir} -> {new_dataset_root_subdir})')
                shutil.copytree(dataset_root_subdir, new_dataset_root_subdir)

            shm_meta[new_dataset_root_subdir] = shm_data_use_count + 1

            dataset_root_subdir = new_dataset_root_subdir

            write_shm_meta(shm_meta)

            _SHM_ACTIVE_PATHS.append(new_dataset_root_subdir)

    if name == 'cifar10':
        dataset = datasets.CIFAR10(root=dataset_root_subdir, train=train,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=True)
    elif name == 'cifar100':
        dataset = datasets.CIFAR100(root=dataset_root_subdir, train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=True)
    elif name == 'stl10':
        dataset = datasets.STL10(root=dataset_root_subdir,
                                 split='train' if train else 'test',
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=True)
    elif name == 'pets':
        dataset = datasets.OxfordIIITPet(root=dataset_root_subdir,
                                         split='trainval' if train else 'test',
                                         transform=transform,
                                         target_transform=target_transform,
                                         download=True)
    elif name == 'flowers':
        dataset = datasets.Flowers102(root=dataset_root_subdir,
                                      split='train' if train else 'test',
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=True)
    elif name == 'aircraft':
        dataset = datasets.FGVCAircraft(root=dataset_root_subdir,
                                        split='trainval' if train else 'test',
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=True)
    elif name == 'food':
        dataset = datasets.Food101(root=dataset_root_subdir,
                                   split='train' if train else 'test',
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=True)
    elif name == 'eurosat':
        if train:
            raise ValueError('EuroSAT does not have a train split.')
        dataset = datasets.EuroSAT(root=dataset_root_subdir,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        dataset = datasets.ImageNet(
            root=dataset_root_subdir,
            transform=transform,
            target_transform=target_transform,
            split='train' if train else 'val',
        )
    elif name == 'objectnet':
        base = ObjectNetBase(transform, dataset_root_subdir)
        dataset = base.get_test_dataset()
    elif name == 'caltech101':
        if train:
            raise ValueError('Caltech101 does not have a train split.')
        dataset = datasets.Caltech101(root=dataset_root_subdir,
                                      target_type='category',
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=True)
    elif name == 'mnist':
        dataset = datasets.MNIST(root=dataset_root_subdir, train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=True)
    elif name == 'cub200':
        dataset = CUB200_2011(root=dataset_root_subdir, train=train,
                              transform=transform, crop_to_bbox=True,
                              target_transform=target_transform)
    else:
        raise ValueError(f'Dataset {name} not supported.')

    return dataset
