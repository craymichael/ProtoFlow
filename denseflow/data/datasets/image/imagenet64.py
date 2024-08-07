import os
import torch
import torch.utils.data as data
import numpy as np
import errno
import tarfile
from PIL import Image
from denseflow.data import DATA_PATH


class ImageNet64Dataset(data.Dataset):
    """
    The ImageNet dataset of
    (Russakovsky et al., 2015): https://arxiv.org/abs/1409.0575
    downscaled to 64x64, as used in
    (van den Oord et al., 2016): https://arxiv.org/abs/1601.06759

    Setup the dataset follow the following instructions:
    mkdir -p ./data/imagenet64/{raw,processed}
    cp ./make_imagenet_data.py ./data/imagenet64/processed/
    cd ./data/imagenet64/raw
    wget https://image-net.org/data/downsample/Imagenet64_train_part1.zip
    wget https://image-net.org/data/downsample/Imagenet64_train_part2.zip
    wget https://image-net.org/data/downsample/Imagenet64_val.zip
    unzip Imagenet64_train_part1.zip
    unzip Imagenet64_train_part2.zip
    unzip Imagenet64_val.zip
    cd ../processed
    mkdir -p ./{train_64x64,valid_64x64}
    python make_imagenet_data.py --img_size 64

    """

    urls = [
        'http://image-net.org/small/train_64x64.tar',
        'http://image-net.org/small/valid_64x64.tar'
    ]
    raw_folder = 'imagenet64/raw'
    processed_folder = 'imagenet64/processed'
    train_folder = 'train_64x64'
    valid_folder = 'valid_64x64'

    def __init__(self, root=DATA_PATH, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform

        # if not self._check_raw():
        #     if download:
        #         self.download()
        #     else:
        #         raise RuntimeError('Dataset not found.' +
        #                            ' You can use download=True to download it')
        #
        # if not self._check_processed():
        #     self.process()

        if self.train:
            # self.files = [os.path.join(self.processed_train_folder, file) for file in os.listdir(self.processed_train_folder)]
            self.files = [os.path.join(self.processed_train_folder, subdir, file) for subdir in os.listdir(self.processed_train_folder) for file in os.listdir(os.path.join(self.processed_train_folder, subdir))]
        else:
            # self.files = [os.path.join(self.processed_valid_folder, file) for file in os.listdir(self.processed_valid_folder)]
            self.files = [os.path.join(self.processed_valid_folder, subdir, file) for subdir in os.listdir(self.processed_valid_folder) for file in os.listdir(os.path.join(self.processed_valid_folder, subdir))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tensor: image
        """

        img = Image.open(self.files[index])

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)

    @property
    def raw_file_paths(self):
        return [os.path.join(self.root, self.raw_folder, url.rpartition('/')[2]) for url in self.urls]

    @property
    def processed_data_folder(self):
        return os.path.join(self.root, self.processed_folder)

    @property
    def processed_train_folder(self):
        return os.path.join(self.processed_data_folder, self.train_folder)

    @property
    def processed_valid_folder(self):
        return os.path.join(self.processed_data_folder, self.valid_folder)

    def _check_processed(self):
        return os.path.exists(self.processed_train_folder) and os.path.exists(self.processed_valid_folder)

    def _check_raw(self):
        return os.path.exists(self.raw_file_paths[0]) and os.path.exists(self.raw_file_paths[1])

    def download(self):
        """Download the data if it doesn't exist in processed_folder already."""
        from six.moves import urllib

        if self._check_raw():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url, file_path in zip(self.urls, self.raw_file_paths):
            print('Downloading ' + url)
            urllib.request.urlretrieve(url, file_path)

    def process(self):

        print("Extracting training data...")
        tar = tarfile.open(self.raw_file_paths[0])
        tar.extractall(self.processed_data_folder)
        tar.close()

        print("Extracting validation data...")
        tar = tarfile.open(self.raw_file_paths[1])
        tar.extractall(self.processed_data_folder)
        tar.close()
