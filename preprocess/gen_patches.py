#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import warnings

import openslide
import pandas as pd
import skimage.io as io

import preprocess.view_utils as view_utils
from preprocess.patches_mean_std import get_mean_and_std
from utils.patches import *

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser(description='PAIP2020 Patch Generator')
    parser.add_argument("--patch_len", type=int, default=256, help="Patch size: patch_len x patch_len")
    parser.add_argument("--stride_len", type=int, default=-1, help="Stride of sliding window. Default patch_len // 4")
    parser.add_argument("--slide_level", type=int, default=2, help="Which level dimension")
    args = parser.parse_args()
    return args


args = set_args()

if os.environ.get('LVSC_DATA_PATH') is not None:
    PAIP2020_DATA_PATH = os.environ.get('PAIP2020_DATA_PATH')
else:
    assert False, "Please set the environment variable PAIP2020_DATA_PATH. Read the README!"


def classify_mask(mask):
    """
    Mask categorization depending on their values
    :param mask: Mask image
    :return: (string)
        'background' if no tumour label in the mask
        'tumour' if only tumour label in the mask
        'border' if tumour and background label in the mask
    """
    if (np.unique(mask) == np.array([0])).all():
        return "background"
    elif (np.unique(mask) == np.array([1])).all():
        return "tumour"
    elif (np.unique(mask) == np.array([0, 1])).all():
        return "border"
    else:
        assert False, "Unknown mask status. Values: {}".format(np.unique(mask))


print("-- SETTINGS --")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

patch_len = args.patch_len
if args.stride_len == -1:
    stride_len = patch_len // 4
else:
    stride_len = args.stride_len
p_level = args.slide_level

train_csv = pd.read_csv("utils/data/train.csv")

base_patch_dir = os.path.join(
    PAIP2020_DATA_PATH,
    "Train/patches_level{}_len{}_stride{}".format(p_level, patch_len, stride_len)
)

total_count_patch_types = {"background": 0, "tumour": 0, "border": 0}

for case in range(len(train_csv)):
    raw_name = train_csv.iloc[case]["wsi"].split("/")[-1][:-4]
    print("\n--- Generatin Patches for {} ({}/{})---".format(raw_name, case + 1, len(train_csv)))
    cur_slide = PAIP2020_DATA_PATH + train_csv.iloc[case]["wsi"]
    cur_mask = PAIP2020_DATA_PATH + train_csv.iloc[case]["annotation_tif"]

    mask_img = view_utils.mask_loader(cur_mask, verbose=False)
    print("Mask shape: {}".format(mask_img.shape))

    wsi_head = openslide.OpenSlide(cur_slide)

    pred_h, pred_w = (wsi_head.level_dimensions[p_level][1], wsi_head.level_dimensions[p_level][0])

    slide_img = wsi_head.read_region((0, 0), p_level, wsi_head.level_dimensions[p_level])
    slide_img = np.asarray(slide_img)[:, :, :3]  # Quitamos el canal alpha ya que no tiene información relevante
    print("Image shape: {}".format(slide_img.shape))

    coors_arr = wsi_stride_splitting(pred_h, pred_w, patch_len=patch_len, stride_len=stride_len)
    print("Total number of patches: {}.".format(len(coors_arr)))

    patch_dir = os.path.join(base_patch_dir, raw_name)
    os.makedirs(patch_dir, exist_ok=True)

    total_patch_types = {"background": 0, "tumour": 0, "border": 0}
    for coor in coors_arr:
        ph, pw = coor[0], coor[1]
        tmp_slide = slide_img[ph:ph + patch_len, pw:pw + patch_len]
        tmp_mask = mask_img[ph:ph + patch_len, pw:pw + patch_len]
        patch_type = classify_mask(tmp_mask)
        total_patch_types[patch_type] += 1
        total_count_patch_types[patch_type] += 1
        patch_name = "ph{}_pw{}_{}".format(ph, pw, patch_type)

        io.imsave(os.path.join(patch_dir, patch_name + ".jpg"), tmp_slide)
        io.imsave(os.path.join(patch_dir, patch_name + ".png"), tmp_mask)

    print("Patches Info: {}".format(total_patch_types))

print("\n--------------------------------------------------")
print("All Patches Info: {}".format(total_count_patch_types))
print("\nGenerating patches csv info...")

train_info = []
indx = 0

for subdir, dirs, files in list(os.walk(base_patch_dir)):
    for file in files:
        if file.endswith(".jpg"):  # Mask have same path but .png extension
            relative_path_patch = os.path.join(
                "/".join(subdir.split("/")[-3:]),
                file
            )
            case = subdir.split("/")[-1]

            train_info.append({
                "case": case,
                "patch": relative_path_patch,
                "mask": relative_path_patch[:-3] + "png",
                "type": file.split("_")[-1][:-4],
                "ph": file[file.find("ph") + 2:file.find("_pw")],
                "pw": file[file.find("pw") + 2:file.rfind("_")],
                "MSI-H": train_csv.loc[train_csv["case"] == case]["MSI-H"].item()
            })

train_info = pd.DataFrame(train_info)
train_info.to_csv("utils/data/patches_level{}_len{}_stride{}.csv".format(p_level, patch_len, stride_len), index=False)

print("Done!\n\nCalculating mean and std...")

rgb_mean, rgb_std = get_mean_and_std(base_patch_dir, suffix=".jpg")
print("Mean rgb: {}".format(rgb_mean))
print("std rgb: {}".format(rgb_std))
