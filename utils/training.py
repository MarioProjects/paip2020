import os
from time import gmtime, strftime

import cv2
import matplotlib.pyplot as plt
import openslide
import pandas as pd
from torch.utils.data import DataLoader

from utils.dataload import mask_loader, PatchArrayDataset
from utils.losses import *
from utils.metrics import *
from utils.onecyclelr import OneCycleLR
from utils.patches import *
from utils.radam import *


def current_time():
    """
    Gives current time
    :return: (String) Current time formated %Y-%m-%d %H:%M:%S
    """
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def dict2df(my_dict, path):
    """
    Save python dictionary as csv using pandas dataframe
    :param my_dict: Dictionary like {"epoch": [1, 2], "accuracy": [0.5, 0.9]}
    :param path: /path/to/file.csv
    :return: (void) Save csv on specified path
    """
    df = pd.DataFrame.from_dict(my_dict, orient="columns")
    df.index.names = ['epoch']
    df.to_csv(path, index=True)


def reescale_img(img, dim):
    """
    Reescale image / Util for img to low resolution and original resolution via padding with 0s
    :param img: (np.array) Image to reescale with padding/unpadding (height, width, channels)
    :param dim: (tuple) Desired final img dimension (height, width)
    :return: Reescaled img to desired dim using padding and unpadding with 0s
    """
    if dim[0] < img.shape[0] or dim[1] < img.shape[1]:  # Downsample -> Padding + Resize /// IMGS
        # First create a squared zeros egion where will we put the image
        max_size = max(img.shape[:2])
        padded_array = np.zeros((max_size, max_size, 3)).astype(img.dtype)
        # Place the image at coresponding place/coordinates
        shape = np.shape(img)
        padded_array[:shape[0], :shape[1], :shape[2]] = img
        # Finally reescale the image without loosing initial image aspect ratio
        return cv2.resize(padded_array, dim)
    else:  # Upsample -> Unpadding + Resize /// MASKS
        # Resize image to original max shape (respect aspect ratio)
        max_size = max(dim)
        upsampled = cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_AREA)
        # Place upsampled image and return
        unpadded_array = upsampled[:dim[0], :dim[1]]
        return unpadded_array


def get_current_lr(optimizer):
    """
    Gives the current learning rate of optimizer
    :param optimizer: Optimizer instance
    :return: Learning rate of specified optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(optmizer_type, model, lr=0.1):
    """
    Create an instance of optimizer
    :param optmizer_type: (string) Optimizer name
    :param model: Model that optimizer will use
    :param lr: Learning rate
    :return: Instance of specified optimizer
    """
    if optmizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif optmizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optmizer_type == "over9000":
        return Over9000(filter(lambda p: p.requires_grad, model.parameters()))

    assert False, "No optimizer named: {}".format(optmizer_type)


def get_scheduler(scheduler_name, optimizer, epochs=40, min_lr=0.002, max_lr=0.01, scheduler_steps=None):
    """
    Gives the specified learning rate scheduler
    :param scheduler_name: Scheduler name
    :param optimizer: Optimizer which is changed by the scheduler
    :param epochs: Total training epochs
    :param min_lr: Minimum learning rate for OneCycleLR Scheduler
    :param max_lr: Maximum learning rate for OneCycleLR Scheduler
    :param scheduler_steps: If scheduler steps is selected, which steps perform
    :return: Instance of learning rate scheduler
    """
    if scheduler_name == "steps":
        if scheduler_steps is None:
            assert False, "Please specify scheduler steps."
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.1)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=6, factor=0.1, patience=12)
    elif scheduler_name == "one_cycle_lr":
        return OneCycleLR(optimizer, num_steps=epochs, lr_range=(min_lr, max_lr))
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999], gamma=0.1)
    else:
        assert False, "Unknown scheduler: {}".format(scheduler_name)


def scheduler_step(optimizer, scheduler, metric, args):
    """
    Perform a step of a scheduler
    :param optimizer: Optimizer used during training
    :param scheduler: Scheduler instance
    :param metric: Metric to minimize
    :param args: Training list of arguments with required fields (Bool: apply_swa, String: scheduler_name)
    :return: (void) Apply scheduler step
    """
    if args.apply_swa:
        optimizer.step()
    if args.scheduler == "steps":
        scheduler.step()
    elif args.scheduler == "plateau":
        scheduler.step(metric)
    elif args.scheduler == "one_cycle_lr":
        scheduler.step()
    elif args.scheduler == "constant":
        pass  # No modify learning rate


def get_criterion(criterion_type, weights_criterion='default'):
    """
    Gives a list of subcriterions and their corresponding weight
    :param criterion_type: Name of created criterion
    :param weights_criterion: (optional) Weight for each subcriterion
    :return:
        (list) Subcriterions
        (list) Weights for each criterion
    """
    if criterion_type == "bce_dice":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion = [criterion1, criterion2, criterion3]
        default_weights_criterion = [0.55, 0.35, 0.1]
    elif criterion_type == "bce_dice_border":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        default_weights_criterion = [0.5, 0.2, 0.1, 0.2]
    elif criterion_type == "bce_dice_ac":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = ActiveContourLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        default_weights_criterion = [0.3, 0.4, 0.2, 0.3]
    else:
        assert False, "Unknown criterion: {}".format(criterion_type)

    if weights_criterion == "default":
        return criterion, default_weights_criterion
    else:
        weights_criterion = [float(i) for i in weights_criterion.split(',')]
        if len(weights_criterion) != len(default_weights_criterion):
            assert False, "We need a weight for each subcriterion"
        return criterion, weights_criterion


def defrost_model(model):
    """
    Unfreeze model parameters
    :param model: Instance of model
    :return: (void)
    """
    for param in model.parameters():  # Defrost model
        param.requires_grad = True


def check_defrost(model, defrosted, current_epoch, args):
    """
    Defrost model if given conditions
    :param model: (Pytorch model) Model to defrost
    :param defrosted: (bool) Current model status. Defrosted (True) or Not (False)
    :param current_epoch: (int) Current training epoch
    :param args: Training list of arguments with required fields (int: defrost_epoch, String: model_name)
    :return: (bool) True if model is defrosted or contrary False
    """
    if not defrosted and current_epoch >= args.defrost_epoch and "scratch" not in args.model_name:
        print("\n---- Unfreeze Model Weights! ----")
        defrost_model(model)
        defrosted = True
    return defrosted


def calculate_loss(y_true, y_pred, criterion, weights_criterion):
    """
    Calculate the loss of generated predictions
    :param y_true: Expected prediction values
    :param y_pred: Model logits
    :param criterion: (list) Criterions to apply
    :param weights_criterion: (list) Weights for each subcriterion
    :return: Loss given by the criterions
    """

    loss = 0
    for indx, crit in enumerate(criterion):
        loss += (weights_criterion[indx] * crit(y_pred, y_true))

    return loss


def train_step(train_loader, model, criterion, weights_criterion, optimizer):
    """
    Perform a train step
    :param train_loader: (Dataset) Train dataset loader
    :param model: Model to train
    :param criterion: Choosed criterion
    :param weights_criterion: Choosed criterion weights
    :param optimizer: Choosed optimizer
    :return: Mean train loss
    """
    train_loss = []
    model.train()
    for image, mask in train_loader:
        image = image.type(torch.float).cuda()
        y_pred = model(image)

        mask = mask.cuda()
        y_pred = y_pred.cuda().float()

        loss = calculate_loss(mask, y_pred, criterion, weights_criterion)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())

    return np.mean(train_loss)


def val_step(val_loader, model, criterion, weights_criterion, save_pred, save_path, args):
    """
    Perform val step depending training mode
    :param val_loader: (Dataloader) Validation dataset loader
    :param model: Model used in training
    :param criterion: Choosed criterion
    :param weights_criterion: (list -> float) Choosed criterion weights
    :param save_pred: (bool) If true save mask predictions
    :param save_path: (string) If save_preds then which directory to save mask predictions
    :param args: (dictionary) Training Arguments
    :return: (void)
    """

    if args.training_mode == "patches":

        iou, dice, val_loss = val_step_patches(
            val_loader.dataset, model, criterion, weights_criterion,
            args.binary_threshold, args.batch_size,
            save_preds=save_pred, save_path=save_path
        )

    elif args.training_mode == "low_resolution":

        iou, dice, val_loss = val_step_low_res(
            val_loader, model, criterion, weights_criterion, args.binary_threshold,
            save_preds=save_pred, save_path=save_path
        )

    else:

        assert False, "Unknown training mode: '{}'".format(args.training_mode)

    return iou, dice, val_loss


def val_step_patches(val_dataset, model, criterion, weights_criterion, binary_threshold, batch_size,
                     save_preds=False, save_path=""):
    """
    Perform a validation step
    :param val_dataset: (Dataset) Validation dataset loader
    :param model: Model used in training
    :param criterion: Choosed criterion
    :param weights_criterion: (list -> float) Choosed criterion weights
    :param binary_threshold: (float) Threshold to set class as class 1. Tipically 0.5
    :param batch_size: (int) Batch size for sliding window prediction
    :param save_preds: (bool) If true save mask predictions
    :param save_path: (string) If save_preds then which directory to save mask predictions
    :return: Intersection Over Union and Dice Metrics, Mean validation loss
    """
    ious, dices, val_loss = [], [], []
    model.eval()
    with torch.no_grad():
        slide_names = val_dataset.get_slide_filenames()

        for num, (cur_slide_path, cur_mask_path) in enumerate(slide_names):
            # Load WSI
            mask = mask_loader(cur_mask_path)

            wsi_head = openslide.OpenSlide(cur_slide_path)
            slide_img = wsi_head.read_region(
                (0, 0), val_dataset.slide_level, wsi_head.level_dimensions[val_dataset.slide_level]
            )
            pred_h = wsi_head.level_dimensions[val_dataset.slide_level][1]
            pred_w = wsi_head.level_dimensions[val_dataset.slide_level][0]
            slide_img = np.asarray(slide_img)[:, :, :3]  # Quitamos el canal alpha ya que no tiene información relevante

            coors_arr = wsi_stride_splitting(pred_h, pred_w, val_dataset.patch_len, val_dataset.stride_len)
            patch_arr, wmap = gen_patch_wmap(slide_img, coors_arr, val_dataset.patch_len)

            patch_dset = PatchArrayDataset(patch_arr, val_dataset.transform, val_dataset.img_transform)
            patch_loader = DataLoader(patch_dset, batch_size=batch_size, shuffle=False, drop_last=False)
            pred_map = np.zeros_like(wmap).astype(np.float32)

            for ind, patches in enumerate(patch_loader):
                inputs = patches.cuda()
                with torch.no_grad():
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    preds = torch.squeeze(preds, dim=1).data.cpu().numpy()
                    if (ind + 1) * batch_size <= len(coors_arr):
                        patch_coors = coors_arr[ind * batch_size:(ind + 1) * batch_size]
                    else:
                        patch_coors = coors_arr[ind * batch_size:]
                    for ind_coor, coor in enumerate(patch_coors):
                        ph, pw = coor[0], coor[1]
                        pred_map[ph:ph + val_dataset.patch_len, pw:pw + val_dataset.patch_len] += preds[ind_coor]

            prob_pred = np.divide(pred_map, wmap)
            y_pred = (prob_pred > binary_threshold).astype(np.uint8)

            ious.append(jaccard_coef(mask, y_pred))
            dices.append(dice_coef(mask, y_pred))
            val_loss.append(
                calculate_loss(torch.from_numpy(mask.astype(np.float32)), torch.from_numpy(prob_pred),
                               criterion, weights_criterion)
            )

            if save_preds:
                os.makedirs(save_path, exist_ok=True)
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 16))
                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')
                ax1.imshow(slide_img)
                ax2.imshow(mask, cmap="gray")
                ax3.imshow(y_pred, cmap="gray")
                pred_filename = os.path.join(
                    save_path,
                    "patches_mask_pred_{}.png".format(cur_slide_path.split("/")[-1][:-4]),
                )
                plt.savefig(pred_filename, bbox_inches='tight', pad_inches=0.25, dpi=250)
                plt.close()

    iou = np.array(ious).mean()
    dice = np.array(dices).mean()

    return iou, dice, np.mean(val_loss)


def val_step_low_res(val_loader, model, criterion, weights_criterion, binary_threshold, save_preds=False, save_path=""):
    """
    Perform a validation step
    :param val_loader: (Dataloader) Validation dataset loader
    :param model: Model used in training
    :param criterion: Choosed criterion
    :param weights_criterion: (list -> float) Choosed criterion weights
    :param binary_threshold: (float) Threshold to set class as class 1. Typically 0.5
    :param save_preds: (bool) If true save mask predictions
    :param save_path: (string) If save_preds then which directory to save mask predictions
    :return: Intersection Over Union and Dice Metrics, Mean validation loss
    """
    ious, dices, val_loss = [], [], []
    original_ious, original_dices = [], []  # Comparing whit original masks (resize prediction)
    model.eval()

    with torch.no_grad():
        for sample_indx, (image, mask, original_img, original_mask) in enumerate(val_loader):
            image = image.type(torch.float).cuda()
            prob_pred = model(image)

            mask = mask.cuda()
            prob_pred = prob_pred.cuda().float()

            for indx, single_pred in enumerate(prob_pred):
                # Metrics os resized space
                y_pred_binary = (single_pred.data.cpu().numpy().squeeze() > binary_threshold).astype(np.uint8)
                ious.append(jaccard_coef(mask[indx].data.cpu().numpy(), y_pred_binary))
                dices.append(dice_coef(mask[indx].data.cpu().numpy(), y_pred_binary))

                # Calculate metrics resizing prediction to original mask shape
                current_original_mask = original_mask[indx].data.cpu().numpy().astype(np.uint8)
                y_pred_binary_resized = reescale_img(y_pred_binary, current_original_mask.shape[:2])

                original_ious.append(jaccard_coef(current_original_mask, y_pred_binary_resized))
                original_dices.append(dice_coef(current_original_mask, y_pred_binary_resized))

                if save_preds:
                    cur_slide_path = val_loader.dataset.df_images.loc[sample_indx]["case"]
                    os.makedirs(save_path, exist_ok=True)
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 16))
                    ax1.axis('off')
                    ax2.axis('off')
                    ax3.axis('off')
                    print(cur_slide_path)
                    current_original_img = original_img[indx].data.cpu().numpy().astype(np.uint8)
                    ax1.imshow(current_original_img)
                    ax2.imshow(current_original_mask.squeeze(), cmap="gray")
                    ax3.imshow(y_pred_binary_resized.squeeze(), cmap="gray")
                    pred_filename = os.path.join(
                        save_path,
                        "low_res_mask_pred_{}.png".format(cur_slide_path),
                    )
                    plt.savefig(pred_filename, bbox_inches='tight', pad_inches=0.25, dpi=250)
                    plt.close()

            val_loss.append(calculate_loss(mask, prob_pred, criterion, weights_criterion).item())

    iou = np.array(ious).mean()
    dice = np.array(dices).mean()

    original_iou = np.array(original_ious).mean()
    original_dice = np.array(original_dices).mean()

    return [iou, original_iou], [dice, original_dice], np.mean(val_loss)
