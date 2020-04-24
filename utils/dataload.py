import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import albumentations

IMG_MEAN = (0.765, 0.547, 0.692)
IMG_STD = (0.092, 0.119, 0.098)

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


class PatchDataset(Dataset):
    """
    Dataset for generated train patches.
    """

    def __init__(self, slide_level, patch_len, stride_len, transform, img_transform,
                 balanced=True, normalize=True, seed=42):

        self.base_dir = PAIP2020_DATA_PATH
        self.slide_level = slide_level
        self.patch_len = patch_len
        self.stride_len = stride_len
        self.normalize = normalize

        df = pd.read_csv("utils/data/patches_level{}_len{}_stride{}.csv".format(slide_level, patch_len, stride_len))

        if balanced:
            # Random sample same number of images of 'background', 'border' and 'tumour'
            pass

        self.df = df

        transform_list = [transforms.ToTensor(), ]
        if normalize:
            transform.append(albumentations.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=1))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.df)

    def get_slide_filenames(self):
        # Return lis of lists [[svs filenames absolute path, corresponding mask absolute path], [...]]
        pass

    def __getitem__(self, idx):
        patch_path = os.path.join(self.base_dir, self.df.loc[idx]["patch"])
        mask_path = os.path.join(self.base_dir, self.df.loc[idx]["mask"])

        image = (io.imread(patch_path) / 255.0).astype(np.float32)
        mask = (io.imread(mask_path) / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image = self.transform(image)

        return [image, mask]


class PatchArrayDataset(Dataset):
    """
    Dataset for slide testing. Each would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, patch_arr, mask_arr=None, normalize=True):
        self.patches = patch_arr
        self.masks = mask_arr

        transform_list = [transforms.ToTensor(), ]
        if normalize:
            transform_list.append(transforms.Normalize(IMG_MEAN, IMG_STD))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx, ...]
        if self.transform:
            patch = self.transform(patch)
        if isinstance(self.masks, np.ndarray):
            mask = np.expand_dims(self.masks[idx, ...], axis=0)
            return patch, mask
        else:
            return patch
