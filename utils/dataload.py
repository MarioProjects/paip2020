import os

import albumentations
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

IMG_MEAN = (0.8807554123271034, 0.8396116564828211, 0.8812618820669211)
IMG_STD = (0.06423696785242902, 0.0768305128408967, 0.06220266645576231)

if os.environ.get('LVSC_DATA_PATH') is not None:
    PAIP2020_DATA_PATH = os.environ.get('PAIP2020_DATA_PATH')
else:
    assert False, "Please set the environment variable PAIP2020_DATA_PATH. Read the README!"


def mask_loader(fn, verbose=False):
    """
    This is a simplest loader for the given tif mask labels, which are compressed in 'LZW' format.
    Scikit-image library can automatically decompress and load them on your physical memory.
    """
    assert (os.path.isfile(fn))
    mask = io.imread(fn)
    if verbose:
        print('mask shape:', mask.shape)
    return mask


def apply_augmentations(image, transform, img_transform, mask=None):
    if transform:
        if mask is not None:
            augmented = transform(image=image, mask=mask)
            mask = augmented['mask']
        else:
            augmented = transform(image=image)

        image = augmented['image']

    if img_transform:
        augmented = img_transform(image=image)
        image = augmented['image']

    return image, mask


class PatchDataset(Dataset):
    """
    Dataset for generated train patches.
    """

    def __init__(self, mode, slide_level, patch_len, stride_len, transform, img_transform,
                 samples_per_type=-1, normalize=False, patch_type="all", seed=42):
        """
        Dataset for generated train patches.
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param slide_level: (int) Which WSI level dimension
        :param patch_len: (int) length of the patch image
        :param stride_len: (int) length of the stride
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param samples_per_type: (int) Number samples per patch type
        :param normalize: (bool) Normalize images using global mean and std
        :param patch_type: (string) Patch type in  ["background", "border", "tumour"]
        :param seed: (int) For reproducibility
        """

        self.base_dir = PAIP2020_DATA_PATH
        self.slide_level = slide_level
        self.patch_len = patch_len
        self.stride_len = stride_len
        self.patch_type = patch_type
        self.mode = mode
        self.normalize = normalize

        self.general_info = pd.read_csv("utils/data/train.csv")
        df = pd.read_csv("utils/data/patches_level{}_len{}_stride{}.csv".format(slide_level, patch_len, stride_len))

        if mode == "train":
            df = df.loc[df["is_validation"] == 0]
        elif mode == "validation":
            df = df.loc[df["is_validation"] == 1]
        else:
            assert False, "Unknown mode '{}'".format(mode)

        if samples_per_type > 0:
            # Random sample same number of images of 'background', 'border' and 'tumour'
            if patch_type != "all":
                assert False, "Not possible samples_per_type and patch type != all"
            df['frequency'] = df['type'].map(1 - df['type'].value_counts() / len(df))
            df = df.sample(samples_per_type * 3, weights=df['frequency'], random_state=seed)  # *3 due have 3 mask types

        if patch_type != "all":
            if patch_type not in ["background", "border", "tumour"]:
                assert False, "Uknown patch type '{}'".format(patch_type)
            # Select only desired patch type
            df = df.loc[df["type"] == patch_type]

        df = df.reset_index(drop=True)
        if self.mode == "train":
            print("\n-- Dataloader statistics: {} --".format(self.mode))
            for indx, row in df.groupby(["type"]).count().iterrows():
                print("'{}' -> {}".format(indx, row["case"]))
        self.df_patches = df

        if normalize:
            img_transform.append(albumentations.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=1))

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return len(self.df_patches)

    def get_slide_filenames(self):
        # Return lis of lists [(svs filenames absolute path, corresponding mask absolute path), (...)]
        selected_cases = self.general_info.loc[self.general_info["case"].isin(self.df_patches["case"].unique())]
        return list(zip(self.base_dir + selected_cases.wsi, self.base_dir + selected_cases.annotation_tif))

    def __getitem__(self, idx):
        patch_path = os.path.join(self.base_dir, self.df_patches.loc[idx]["patch"])
        mask_path = os.path.join(self.base_dir, self.df_patches.loc[idx]["mask"])

        image = (io.imread(patch_path) / 255.0).astype(np.float32)
        mask = io.imread(mask_path).astype(np.float32)

        image, mask = apply_augmentations(image, self.transform, self.img_transform, mask)

        image = torch.from_numpy(image.transpose(2, 0, 1))  # Transpose == Channels first
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float()

        return [image, mask]


class PatchArrayDataset(Dataset):
    """
    Dataset for slide testing. Each would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, patch_arr, transform, img_transform, mask_arr=None, normalize=False):
        """
        Dataset for slide testing
        :param patch_arr: Array of patches
        :param transform: (list) List of albumentations applied to image and mask
        :param img_transform: (list) List of albumentations applied to image only
        :param mask_arr: Optional array of masks
        :param normalize: (bool) Normalize images using global mean and std
        """
        self.patches = patch_arr
        self.masks = mask_arr

        if normalize:
            img_transform.append(albumentations.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=1))

        self.transform = albumentations.Compose(transform)
        self.img_transform = albumentations.Compose(img_transform)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx, ...]
        mask = self.masks[idx, ...] if isinstance(self.masks, np.ndarray) else None

        patch, mask = apply_augmentations(patch, self.transform, self.img_transform, mask)

        patch = torch.from_numpy(patch.transpose(2, 0, 1))  # Transpose == Channels first
        mask = torch.from_numpy(np.expand_dims(mask, 0)).float() if isinstance(self.masks, np.ndarray) else None

        if mask is None:
            return patch
        return [patch, mask]
