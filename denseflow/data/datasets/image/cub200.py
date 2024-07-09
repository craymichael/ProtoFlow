import os.path as osp
import pandas as pd
import torch
from torch.utils import data
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.transforms.functional import get_dimensions


class CUB200_2011(data.Dataset):
    base_folder = "CUB_200_2011/images"
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
            self,
            root,
            train=True,
            transform=None,
            loader=default_loader,
            download=True,
            yield_img_id=False,
            yield_orig_shape=False,
            crop_to_bbox=False,
            part_annotations=False,
    ):
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
                "Dataset not found or corrupted. You can use download=True to download it")

    def _load_metadata(self):
        df_images = pd.read_csv(
            osp.join(self.root, "CUB_200_2011", "images.txt"), sep=" ",
            names=["img_id", "filepath"]
        )
        df_image_class_labels = pd.read_csv(
            osp.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ", names=["img_id", "target"]
        )
        df_train_test_split = pd.read_csv(
            osp.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ", names=["img_id", "is_training_img"]
        )

        self.data = df_images.merge(df_image_class_labels, on="img_id")
        self.data = self.data.merge(df_train_test_split, on="img_id")

        if self.crop_to_bbox:
            df_bbox = pd.read_csv(
                osp.join(self.root, "CUB_200_2011", "bounding_boxes.txt"),
                sep=" ", names=["img_id", "x", "y", "w", "h"]
            )
            df_bbox_int = df_bbox.astype(int)
            assert (df_bbox == df_bbox_int).all().all()
            self.data = self.data.merge(df_bbox_int, on="img_id")

        if self.part_annotations:

            df_parts = pd.read_csv(
                osp.join(self.root, "CUB_200_2011", "parts", "part_locs.txt"),
                sep=" ",
                names=["img_id", "part_id", "x", "y", "visible"],
            )
            df_parts_int = df_parts.astype(int)
            assert (df_parts.astype(float) == df_parts_int).all().all()
            df_parts = df_parts_int
            df_parts["part_id_orig"] = df_parts["part_id"]
            df_parts["part_id"] -= 1
            df_parts["part_id"] = df_parts["part_id"].replace(
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
                print(f"{filepath} is not a file...")
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print(
                f"{type(self).__name__} Files already downloaded and " f"verified")
            return True

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(osp.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)
        return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = osp.join(self.root, self.base_folder, sample.filepath)
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
                return sample.img_id, orig_shape, img
            return sample.img_id, img
        elif self.yield_orig_shape:
            return orig_shape, img
        return img
