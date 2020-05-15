# PAIP 2020

## Environment Setup


To use the code, the user needs to set te environment variable to access the data. At your ~/.bashrc add:
```shell script
export PAIP2020_DATA_PATH='/path/to/data/PAIP2020/'
```

Also, the user needs to to pre-install a feew packages:
```shell script
$ pip install -r requirements.txt
```

### Data Preparation

We have to download provided SWI files with their xml mask and unzip them into `PAIP2020_DATA_PATH` creating
for them 'Train/SWI/' and 'Train/annotation/' folders respectively. Now we generate the .tif mask files from the .xml
running `python3 preprocess/xml2mask.py`. This will create a 'Train/mask_img_l2' folder with the generated masks.

#### Patches
We can generate data patches: `python3 preprocess/gen_patches.py`.
```shell script
usage: gen_patches.py [-h] [--patch_len PATCH_LEN] [--stride_len STRIDE_LEN]
                      [--slide_level SLIDE_LEVEL]

PAIP2020 Patch Generator

optional arguments:
  -h, --help            show this help message and exit
  --patch_len PATCH_LEN
                        Patch size: patch_len x patch_len
  --stride_len STRIDE_LEN
                        Stride of sliding window. Default patch_len // 4
  --slide_level SLIDE_LEVEL
                        Which level dimension
```

#### Low resolution
We can generate low resolution images: `python3 preprocess/gen_low_resolution.py`.
```shell script
usage: gen_low_resolution.py [-h] [--img_size IMG_SIZE]
                             [--slide_level SLIDE_LEVEL]

PAIP2020 Patch Generator

optional arguments:
  -h, --help            show this help message and exit
  --img_size IMG_SIZE   Final squared image size
  --slide_level SLIDE_LEVEL
                        From which level dimension resize
```

#### Get overlays

We can get precomputed train overlays between images and mask with `./scripts/get_overlays.sh`. Will create a folder 
with all train cases overlays. 


## Data Description

We have 47 svs images. A whole-slide image is a digital representation of a microscopic slide, 
typically at a very high level of magnification, 20x. As a result of this high magnification, 
whole slide images are typically very large in size. 

A whole-slide image is created by a microscope that scans a slide and combines smaller images into a 
large image. Techniques include combining scanned square tiles into a whole-slide image and combining 
scanned strips into a resulting whole-slide image. 

This is a pyramidal, tiled format, where the massive slide is composed of a large number 
of constituent tiles.

Our images have 4 levels/dimensions as shows next figure.

![WSI Structure](images/wsi_structure.png "WSI Structure")

For example, the image 'training_data_41.svs' has the next levels: 
`((121512, 93068), (30378, 23267), (7594, 5816), (3797, 2908))`

#### Data Visualization

We can use the OpenSlide project to read a variety of whole-slide image formats, 
including the Aperio *.svs slide format of our training image set.

To use the OpenSlide Python interface to view whole slide images, 
we can clone the [OpenSlide Python interface from GitHub](https://github.com/openslide/openslide-python)
and use the included DeepZoom deepzoom_multiserver.py script.

```shell script
$ git clone https://github.com/openslide/openslide-python.git
$ cd openslide-python/examples/deepzoom
$ python3 deepzoom_multiserver.py -Q 100 WSI_DIRECTORY
```

The deepzoom_multiserver.py script starts a web interface on port 5000 and displays the image 
files at the specified file system location (the WSI_DIRECTORY value in the previous code).
If image files exist in subdirectories, they will also be displayed in the list of available slides.

Note: For remote data (ssh) use first:

```shell script
$ ssh -L 5000:localhost:5000 user@host
```

## Results

#### Low Resolution

Calculated using resnet34_unet_imagenet_encoder, Adam and constant learning rate. Using bce_dice 0.1,0.85,0.05.

| Method                             |  Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Average |
|------------------------------------|:-------:|:------:|:------:|:------:|:------:|:-------:|
| 512 - DA None - lr 0.0001          |  0.6677 | 0.6152 | 0.6392 | 0.6984 | 0.6315 |  0.6504 |
| 512 - DA None - lr 0.00001         |  0.6607 | 0.5659 | 0.6006 | 0.6929 | 0.6041 |  0.6248 |
| 512 - DA Rotation - lr 0.0001      |  0.7162 | 0.6336 | 0.6663 | 0.7496 | 0.6641 |  0.6859 |
| 512 - DA Rotation - lr 0.00001     |  0.6897 | 0.6051 | 0.6305 | 0.7107 | 0.6332 |  0.6534 |
| 512 - DA VFlip - lr 0.0001         |  0.7170 | 0.6312 | 0.6400 | 0.7471 | 0.6536 |  0.6777 |
| 512 - DA VFlip - lr 0.00001        |  0.6824 | 0.5861 | 0.6283 | 0.6940 | 0.6232 |  0.6427 |
| 512 - DA HFlip - lr 0.0001         |  0.7031 | 0.6436 | 0.6476 | 0.7399 | 0.6387 |  0.6745 |
| 512 - DA HFlip - lr 0.00001        |  0.6787 | 0.5943 | 0.6321 | 0.7124 | 0.6062 |  0.6447 |
| 512 - DA Elastic - lr 0.0001       |  0.6987 | 0.6744 | 0.6149 | 0.7295 | 0.6277 |  0.6690 |
| 512 - DA Elastic - lr 0.00001      |  0.6782 | 0.5722 | 0.6149 | 0.7121 | 0.6362 |  0.6427 |
| 512 - DA Grid Dist - lr 0.0001     |  0.7372 | 0.6476 | 0.6529 | 0.7362 | 0.6341 |  0.6816 |
| 512 - DA Grid Dist - lr 0.00001    |  0.7394 | 0.6185 | 0.6161 | 0.6966 | 0.6154 |  0.6571 |
| 512 - DA Shift - lr 0.0001         |  0.7138 | 0.6702 | 0.6503 | 0.7437 | 0.6408 |  0.6837 |
| 512 - DA Shift - lr 0.00001        |  0.7050 | 0.6000 | 0.6053 | 0.7126 | 0.6210 |  0.6487 |
| 512 - DA Scale - lr 0.0001         |  0.7217 | 0.6337 | 0.6537 | 0.7285 | 0.6602 |  0.6795 |
| 512 - DA Scale - lr 0.00001        |  0.6954 | 0.6065 | 0.6312 | 0.6924 | 0.6277 |  0.6506 |
| 512 - DA Opt Dist - lr 0.0001      |  0.7377 | 0.5934 | 0.6266 | 0.7288 | 0.6272 |  0.6627 |
| 512 - DA Opt Dist - lr 0.00001     |  0.6858 | 0.5964 | 0.6190 | 0.6889 | 0.6183 |  0.6416 |
| 512 - DA Cutout - lr 0.0001        |  0.6769 | 0.6144 | 0.6367 | 0.7029 | 0.6394 |  0.6540 |
| 512 - DA Cutout - lr 0.00001       |  0.6561 | 0.5707 | 0.6209 | 0.6949 | 0.6331 |  0.6351 |
| 512 - DA Random Crops - lr 0.0001  |  0.5901 | 0.5964 | 0.5521 | 0.7476 | 0.6563 |  0.6285 |
| 512 - DA Random Crops - lr 0.00001 |  0.5850 | 0.5043 | 0.5369 | 0.5952 | 0.6357 |  0.5710 |
| 512 - DA Downscale - lr 0.0001     |  0.7066 | 0.5634 | 0.5901 | 0.5864 | 0.6253 |  0.6416 |
| 512 - DA Downscale - lr 0.00001    |  0.6572 | 0.5546 | 0.6240 | 0.7055 | 0.6145 |  0.6311 |
| 512 - DA Combination - lr 0.0001   |  0.7226 | 0.7074 | 0.6478 | 0.7023 | 0.6668 |  0.6893 |
| 512 - DA Combination - lr 0.00001  |  0.6837 | 0.5728 | 0.6222 | 0.6708 | 0.6141 |  0.6327 |

Transformations that seem to work: Rotations, Flips, Scale, Shift. Using scratch (**random weights**) model.
Using bce_dice 0.1,0.85,0.05.

| Method                             |  Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Average |
|------------------------------------|:-------:|:------:|:------:|:------:|:------:|:-------:|
| 512 - DA None - lr 0.0001          |  0.6385 | 0.5542 | 0.6138 | 0.6530 | 0.5963 |  0.6111 |
| 512 - DA Combination - lr 0.0001   |  0.6962 | 0.5559 | 0.6065 | 0.6633 | 0.5901 |  0.6224 |

With smaller **random** Unet. Using bce_dice 0.1,0.85,0.05.

| Method                             |  Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Average |
|------------------------------------|:-------:|:------:|:------:|:------:|:------:|:-------:|
| 512 - DA None - lr 0.01            |  0.6492 | 0.5893 | 0.6523 | 0.6853 | 0.5914 |  0.6335 |
| 512 - DA None - lr 0.001           |  0.7037 | 0.6972 | 0.6671 | 0.7768 | 0.6090 |  0.6907 |
| 512 - DA None - lr 0.0001          |  0.6464 | 0.6318 | 0.6814 | 0.6733 | 0.5852 |  0.6436 |
| 512 - DA Combination - lr 0.01     |  0.6695 | 0.6112 | 0.6242 | 0.6753 | 0.5608 |  0.6282 |
| 512 - DA Combination - lr 0.001    |  0.7385 | 0.6215 | 0.6777 | 0.6874 | 0.6031 |  0.6656 |
| 512 - DA Combination - lr 0.0001   |  0.5709 | 0.6182 | 0.6343 | 0.7219 | 0.5967 |  0.6284 |


With smaller **random** Unet, Over9000 OneCycleLR, bce_dice 0.1,0.85,0.05:

| Method               |               LR            |  Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Average |
|----------------------|:---------------------------:|:-------:|:------:|:------:|:------:|:------:|:-------:|
| 512 - DA None        | minlr 0.00001 maxlr 0.001   |  0.6748 | 0.7326 | 0.5257 | 0.6304 | 0.6058 |  0.6338 |
| 512 - DA None        | minlr 0.000001 maxlr 0.0001 |  0.6013 | 0.6549 | 0.4543 | 0.6264 | 0.5953 |  0.5864 |
| 512 - DA Combination | minlr 0.00001 maxlr 0.001   |  0.5998 | 0.7900 | 0.4434 | 0.7107 | 0.5821 |  0.6252 |
| 512 - DA Combination | minlr 0.000001 maxlr 0.0001 |  0.5592 | 0.5399 | 0.4461 | 0.6165 | 0.5766 |  0.5476 |


## Other

- Paip 2019 site [here](https://paip2019.grand-challenge.org/)
- Paip 2020 Gran Challenge site [here](https://paip2020.grand-challenge.org/Home/)
- Paip 2020 [data](http://wisepaip.org/challenge2020)
- Paip 2019 [GitHub](https://github.com/paip-2019/challenge) & Paip 2020 [GitHub](https://github.com/wisepaip/paip2020)
- [GitHub](https://github.com/PingjunChen/LiverCancerSeg) with data processing Paip 2019 
- IBM [Preprocessing](https://developer.ibm.com/technologies/data-science/articles/an-automatic-method-to-identify-tissues-from-big-whole-slide-images-pt1/)
- Paip 2019 [Best Result](https://drive.google.com/file/d/1NWJdZJFHajHfod5fgO7WnoRqYuQGN_tp/view), [Second](https://drive.google.com/file/d/14Jlv339SXF42vkwlqG5kYW-zGWw5O1Mh/view) and [Third](https://drive.google.com/file/d/1Q5gfmL7SQ_9YINx3J2PR4pQihNyIQxtL/view)