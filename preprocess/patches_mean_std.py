#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pydaily
from skimage import io


def set_args():
    parser = argparse.ArgumentParser(description='PAIP2020 Mean & std Patch Calculator')
    parser.add_argument("--patch_len", type=int, default=256, help="Patch size: patch_len x patch_len")
    parser.add_argument("--stride_len", type=int, default=-1, help="Stride of sliding window. Default patch_len // 4")
    parser.add_argument("--slide_level", type=int, default=2, help="Which level dimension")
    args = parser.parse_args()
    return args


if os.environ.get('LVSC_DATA_PATH') is not None:
    PAIP2020_DATA_PATH = os.environ.get('PAIP2020_DATA_PATH')
else:
    assert False, "Please set the environment variable PAIP2020_DATA_PATH. Read the README!"


def get_mean_and_std(img_dir, suffix):
    mean, std = np.zeros(3), np.zeros(3)
    filelist = pydaily.filesystem.find_ext_files(img_dir, suffix)

    for idx, filepath in enumerate(filelist):
        cur_img = io.imread(filepath) / 255.0
        for i in range(3):
            mean[i] += np.mean(cur_img[:, :, i])
            std[i] += cur_img[:, :, i].std()
    mean = [ele * 1.0 / len(filelist) for ele in mean]
    std = [ele * 1.0 / len(filelist) for ele in std]
    return mean, std


if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    args = set_args()

    patches_dir = os.path.join(
        PAIP2020_DATA_PATH,
        "Train/patches_level{}_len{}_stride{}/".format(args.slide_level, args.patch_len, args.stride_len)
    )

    rgb_mean, rgb_std = get_mean_and_std(patches_dir, suffix=".jpg")
    print("mean rgb: {}".format(rgb_mean))
    print("std rgb: {}".format(rgb_std))
