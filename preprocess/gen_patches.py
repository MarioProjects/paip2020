import argparse
import os
import warnings
from os import environ

import openslide
import pandas as pd
import skimage.io as io

import preprocess.view_utils as view_utils
from utils.patches import *

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser(description='PAIP2020 Patch Generator')
    parser.add_argument("--patch_len", type=int, default=256, help="Patch size: patch_len x patch_len")
    parser.add_argument("--stride_len", type=int, default=-1, help="Stride of sliding window. Default patch_len // 4")
    parser.add_argument("--slide_level", type=int, default=2, help="Which level dimension")
    args = parser.parse_args()
    return args


if environ.get('LVSC_DATA_PATH') is not None:
    PAIP2020_DATA_PATH = environ.get('PAIP2020_DATA_PATH')
else:
    assert False, "Please set the environment variable PAIP2020_DATA_PATH. Read the README!"


def classify_mask(mask):
    if (np.unique(mask) == np.array([0])).all():
        return "background"
    elif (np.unique(mask) == np.array([1])).all():
        return "tumour"
    elif (np.unique(mask) == np.array([0, 1])).all():
        return "border"
    else:
        assert False, "Uknown mask status. Values: {}".format(np.unique(mask))


args = set_args()
print("-- SETTINGS --")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

patch_len = args.patch_len
if args.stride_len == -1:
    stride_len = patch_len // 4
else:
    stride_len = args.stride_len
p_level = args.slide_level

train_info = pd.read_csv("utils/data/train.csv")

for case in range(len(train_info)):
    raw_name = train_info.iloc[case]["wsi"].split("/")[-1][:-4]
    print("\n--- Generatin Patches for {} ({}/{})---".format(raw_name, case + 1, len(train_info)))
    cur_slide = PAIP2020_DATA_PATH + train_info.iloc[case]["wsi"]
    cur_mask = PAIP2020_DATA_PATH + train_info.iloc[case]["annotation_tif"]

    mask_img = view_utils.mask_loader(cur_mask, verbose=False)
    print("Mask shape: {}".format(mask_img.shape))

    wsi_head = openslide.OpenSlide(cur_slide)

    pred_h, pred_w = (wsi_head.level_dimensions[p_level][1], wsi_head.level_dimensions[p_level][0])

    slide_img = wsi_head.read_region((0, 0), p_level, wsi_head.level_dimensions[p_level])
    slide_img = np.asarray(slide_img)[:, :, :3]  # Quitamos el canal alpha ya que no tiene información relevante
    print("Image shape: {}".format(slide_img.shape))

    coors_arr = wsi_stride_splitting(pred_h, pred_w, patch_len=patch_len, stride_len=stride_len)
    print("Total number of patches: {}.".format(len(coors_arr)))

    patch_dir = PAIP2020_DATA_PATH + "Train/patches_len{}_stride{}/{}".format(patch_len, stride_len, raw_name)
    os.makedirs(patch_dir, exist_ok=True)

    total_patch_types = {"background": 0, "tumour": 0, "border": 0}
    for coor in coors_arr:
        ph, pw = coor[0], coor[1]
        tmp_slide = slide_img[ph:ph + patch_len, pw:pw + patch_len]
        tmp_mask = mask_img[ph:ph + patch_len, pw:pw + patch_len]
        patch_type = classify_mask(tmp_mask)
        total_patch_types[patch_type] += 1
        patch_name = "ph{}_pw{}_{}".format(ph, pw, patch_type)

        io.imsave(os.path.join(patch_dir, patch_name + ".jpg"), tmp_slide)
        io.imsave(os.path.join(patch_dir, patch_name + ".png"), tmp_mask)

    print("Patches Info: {}".format(total_patch_types))
