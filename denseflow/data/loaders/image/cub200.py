from denseflow.data.datasets.image import CUB200_2011
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from denseflow.data.transforms import Quantize
from denseflow.data import TrainTestLoader, DATA_PATH


class CUB200(TrainTestLoader):

    def __init__(self, root=DATA_PATH, download=True, num_bits=8,
                 pil_transforms=[]):
        self.root = root

        resize_trans = [Resize(64), CenterCrop(64)]
        trans_train = resize_trans + pil_transforms + [ToTensor(),
                                                       Quantize(num_bits)]
        trans_test = resize_trans + [ToTensor(), Quantize(num_bits)]

        self.train = CUB200_2011(root, train=True,
                                 transform=Compose(trans_train),
                                 download=download, crop_to_bbox=True)
        self.test = CUB200_2011(root, train=False,
                                transform=Compose(trans_test),
                                crop_to_bbox=True)
