import os
import json
import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='PAIP 2020 Challenge - CRC Prediction', formatter_class=SmartFormatter)

parser.add_argument('--verbose', action='store_true', help='Verbose mode')
parser.add_argument('--output_dir', type=str, help='Where progress/checkpoints will be saved')

parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')

parser.add_argument('--model_name', type=str, default='simple_unet', help='Model name for training')
parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--crop_size', type=int, default=200, help='Center crop squared size')
parser.add_argument('--img_size', type=int, default=200, help='Final img squared size')

parser.add_argument('--default_threshold', type=float, default=0.5, help='Default threshold at training time')

parser.add_argument('--criterion', type=str, default='bce', help='Criterion for training')
parser.add_argument('--weights_criterion', type=str, default='default', help='Weights for each subcriterion')

parser.add_argument('--model_checkpoint', type=str, default="", help='Where is the model checkpoint saved')
parser.add_argument('--defrost_epoch', type=int, default=-1, help='Number of epochs to defrost the model')

parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
parser.add_argument('--scheduler_name', type=str, default="", help='Where is the model checkpoint saved')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--min_learning_rate', type=float, default=0.000001, help='Minimun Learning rate')
parser.add_argument('--max_learning_rate', type=float, default=0.000001, help='Maximum Learning rate')
parser.add_argument('--scheduler_steps', '--arg', nargs='+', type=int, help='Steps when steps scheduler choosed')

parser.add_argument('--apply_swa', action='store_true', help='Apply stochastic weight averaging')
parser.add_argument('--swa_freq', type=int, default=1, help='SWA Frequency')
parser.add_argument('--swa_start', type=int, default=60, help='SWA_LR')
parser.add_argument('--swa_lr', type=float, default=0.0001, help='SWA_LR')

parser.add_argument('--progress_imgs', action='store_true', help='Create image predictions every epoch')
parser.add_argument('--plot_filters', action='store_true', help='Plot/Save initial filters every epoch')

args = parser.parse_args()

# try:
#     args = parser.parse_args()
# except:
#     print("Working with Jupyter notebook! (Default Arguments)")
#     args = parser.parse_args("")

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# https://stackoverflow.com/a/55114771
with open(args.output_dir + '/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
