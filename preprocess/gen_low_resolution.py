#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import warnings

import cv2
import openslide
import pandas as pd
import skimage.io as io

import preprocess.view_utils as view_utils
from utils.patches import *

warnings.filterwarnings('ignore')


def set_args():
    parser = argparse.ArgumentParser(description='PAIP2020 Patch Generator')
    parser.add_argument("--img_size", type=int, default=512, help="Final squared image size")
    parser.add_argument("--slide_level", type=int, default=2, help="From which level dimension resize")
    args = parser.parse_args()
    return args


args = set_args()

if os.environ.get('LVSC_DATA_PATH') is not None:
    PAIP2020_DATA_PATH = os.environ.get('PAIP2020_DATA_PATH')
else:
    assert False, "Please set the environment variable PAIP2020_DATA_PATH. Read the README!"

print("-- SETTINGS --")
for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

train_csv = pd.read_csv("utils/data/train.csv")

base_resize_dir = os.path.join(
    PAIP2020_DATA_PATH,
    "Train/resized_level{}_size{}".format(args.slide_level, args.img_size)
)

os.makedirs(base_resize_dir, exist_ok=True)

total_count_patch_types = {"background": 0, "tumour": 0, "border": 0}

for case in range(len(train_csv)):
    raw_name = train_csv.iloc[case]["wsi"].split("/")[-1][:-4]
    print("\n--- Generatin Resized Image for {} ({}/{})---".format(raw_name, case + 1, len(train_csv)))
    cur_slide = PAIP2020_DATA_PATH + train_csv.iloc[case]["wsi"]
    cur_mask = PAIP2020_DATA_PATH + train_csv.iloc[case]["annotation_tif"]

    mask_img = view_utils.mask_loader(cur_mask, verbose=False)
    print("Mask shape: {}".format(mask_img.shape))

    wsi_head = openslide.OpenSlide(cur_slide)

    pred_h, pred_w = (wsi_head.level_dimensions[args.slide_level][1], wsi_head.level_dimensions[args.slide_level][0])

    slide_img = wsi_head.read_region((0, 0), args.slide_level, wsi_head.level_dimensions[args.slide_level])
    slide_img = np.asarray(slide_img)[:, :, :3]  # Quitamos el canal alpha ya que no tiene información relevante
    print("Image shape: {}".format(slide_img.shape))

    max_size = max(slide_img.shape[:2])
    resize_dim = (args.img_size, args.img_size)
    original_shape = np.shape(slide_img)

    # Resize image
    padded_img = np.zeros((max_size, max_size, 3)).astype(slide_img.dtype)

    padded_img[:original_shape[0], :original_shape[1], :original_shape[2]] = slide_img

    resized_img = cv2.resize(padded_img, resize_dim)

    # Resize mask
    padded_mask = np.zeros((max_size, max_size)).astype(mask_img.dtype)

    padded_mask[:original_shape[0], :original_shape[1]] = mask_img

    resized_mask = cv2.resize(padded_mask, resize_dim)

    # Save resized result
    io.imsave(os.path.join(base_resize_dir, raw_name + ".jpg"), resized_img)
    io.imsave(os.path.join(base_resize_dir, raw_name + ".png"), resized_mask)

train_info = []
indx = 0

for subdir, dirs, files in list(os.walk(base_resize_dir)):
    for file in files:
        if file.endswith(".jpg"):  # Mask have same path but .png extension
            relative_path_img = os.path.join(
                "/".join(subdir.split("/")[-2:]),
                file
            )
            case = relative_path_img.split("/")[-1][:-4]

            train_info.append({
                "case": case,
                "image": relative_path_img,
                "mask": relative_path_img[:-3] + "png",
                "MSI-H": train_csv.loc[train_csv["case"] == case]["MSI-H"].item()
            })

train_info = pd.DataFrame(train_info)
train_info.to_csv("utils/data/resized_level{}_size{}.csv".format(args.slide_level, args.img_size), index=False)

print("Done!")
