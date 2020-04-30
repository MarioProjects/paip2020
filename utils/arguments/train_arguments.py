import argparse
import json
import os


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='PAIP 2020 Challenge - CRC Prediction', formatter_class=SmartFormatter)

parser.add_argument("--gpu", type=str, default="0,1")
parser.add_argument("--seed", type=int, default=2020)
parser.add_argument('--output_dir', type=str, help='Where progress/checkpoints will be saved')

parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size for training')

parser.add_argument('--model_name', type=str, default='simple_unet', help='Model name for training')
parser.add_argument('--data_augmentation', type=str, help='Apply data augmentations at train time')
parser.add_argument('--crop_size', type=int, default=200, help='Center crop squared size')
parser.add_argument('--img_size', type=int, default=200, help='Final img squared size')

parser.add_argument('--training_mode', type=str, choices=["patches", "low_resolution"], help='How to train the model')
parser.add_argument('--slide_level', type=int, default=2, help='Which WSI level dimension')
parser.add_argument('--patch_len', type=int, default=256, help='Length of the patch image')
parser.add_argument('--stride_len', type=int, default=64, help='Length of the stride')
parser.add_argument('--low_res', type=int, default=512, help='Which image size for low resolution training mode')
parser.add_argument('--samples_per_type', type=int, default=-1, help='Number samples per patch type. Default all')
parser.add_argument('--normalize', action='store_true', help='Normalize images using global mean and std')

parser.add_argument('--binary_threshold', type=float, default=0.5, help='Threshold for masks probabilities')

parser.add_argument('--criterion', type=str, default='bce', help='Criterion for training')
parser.add_argument('--weights_criterion', type=str, default='default', help='Weights for each subcriterion')

parser.add_argument('--model_checkpoint', type=str, default="", help='Where is the model checkpoint saved')
parser.add_argument('--defrost_epoch', type=int, default=-1, help='Number of epochs to defrost the model')

parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training')
parser.add_argument('--scheduler', type=str, default="", help='Where is the model checkpoint saved')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--min_lr', type=float, default=0.0001, help='Minimun Learning rate')
parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum Learning rate')
parser.add_argument('--scheduler_steps', '--arg', nargs='+', type=int, help='Steps when steps scheduler choosed')

parser.add_argument('--apply_swa', action='store_true', help='Apply stochastic weight averaging')
parser.add_argument('--swa_freq', type=int, default=1, help='SWA Frequency')
parser.add_argument('--swa_start', type=int, default=60, help='SWA_LR')
parser.add_argument('--swa_lr', type=float, default=0.0001, help='SWA_LR')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

for argument in args.__dict__:
    print("{}: {}".format(argument, args.__dict__[argument]))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# https://stackoverflow.com/a/55114771
with open(args.output_dir + '/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
