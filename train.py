#!/usr/bin/env python
# coding: utf-8

# ---- Library import ----

from torchcontrib.optim import SWA

# ---- My utils ----
from models import *
from utils.arguments.train_arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.dataload import PatchDataset
from utils.training import *

np.set_printoptions(precision=4)
train_aug, train_aug_img, val_aug = data_augmentation_selector(args.data_augmentation, args.img_size, args.crop_size)

train_dataset = PatchDataset(
    "train", args.slide_level, args.patch_len, args.stride_len, train_aug, train_aug_img,
    normalize=args.normalize, patch_type="all", samples_per_type=args.samples_per_type, seed=args.seed
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

val_dataset = PatchDataset(
    "validation", args.slide_level, args.patch_len, args.stride_len, val_aug, [],
    normalize=args.normalize, patch_type="all", seed=args.seed
)

val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

model = model_selector(args.model_name, in_size=(args.crop_size, args.crop_size))
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
if args.model_checkpoint != "":
    print("Load from pretrained checkpoint: {}".format(args.model_checkpoint))
    model.load_state_dict(torch.load(args.model_checkpoint))

criterion, weights_criterion = get_criterion(args.criterion, args.weights_criterion)

optimizer = get_optimizer(args.optimizer, model, lr=args.learning_rate)
if args.apply_swa:
    optimizer = SWA(optimizer, swa_start=args.swa_start, swa_freq=args.swa_freq, swa_lr=args.swa_lr)

scheduler = get_scheduler(
    args.scheduler, optimizer, epochs=args.epochs,
    min_lr=args.min_lr, max_lr=args.max_lr,
    scheduler_steps=args.scheduler_steps
)

progress = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}
best_iou, best_model, defrosted = -1, None, False

print("\n-------------- START TRAINING -------------- ")
for current_epoch in range(args.epochs):

    defrosted = check_defrost(model, defrosted, current_epoch, args)

    train_loss = train_step(train_loader, model, criterion, weights_criterion, optimizer)

    iou, dice, val_loss = val_step(
        val_dataset, model, criterion, weights_criterion,
        args.binary_threshold, args.batch_size
    )

    print("[" + current_time() + "] Epoch: %d, LR: %.8f, Train: %.6f, Val: %.6f, Val IOU: %s, Val Dice: %s" % (
        current_epoch + 1, get_current_lr(optimizer), train_loss, val_loss, iou, dice))

    if iou > best_iou and not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_best_iou.pt")
        best_iou = iou
        best_model = model.state_dict()

    if not args.apply_swa:
        torch.save(model.state_dict(), args.output_dir + "/model_" + args.model_name + "_last.pt")

    progress["train_loss"].append(np.mean(train_loss))
    progress["val_loss"].append(np.mean(val_loss))
    progress["val_iou"].append(iou)
    progress["val_dice"].append(dice)

    dict2df(progress, args.output_dir + 'progress.csv')

    scheduler_step(optimizer, scheduler, iou, args)

# --------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------- #

if args.apply_swa:
    torch.save(optimizer.state_dict(), args.output_dir + "/optimizer_" + args.model_name + "_before_swa_swap.pt")
    optimizer.swap_swa_sgd()  # Set the weights of your model to their SWA averages
    optimizer.bn_update(train_loader, model, device='cuda')

    torch.save(
        model.state_dict(),
        args.output_dir + "/swa_checkpoint_last_bn_update_{}epochs_lr{}.pt".format(args.epochs, args.swa_lr)
    )

    iou, dice, val_loss = val_step(
        val_dataset, model, criterion, weights_criterion,
        args.binary_threshold, args.batch_size
    )

    print("[SWA] Val IOU: %s, Val Dice: %s" % (iou, dice))

print("\n---------------")
val_iou = np.array(progress["val_iou"])
val_dice = np.array(progress["val_dice"])
print("Best IOU {:.4f} at epoch {}".format(val_iou.max(), val_iou.argmax() + 1))
print("Best DICE {:.4f} at epoch {}".format(val_dice.max(), val_dice.argmax() + 1))
print("---------------\n")
