import os
import json
import torch
from os.path import join
from argparse import ArgumentParser, Namespace
from constant import IMPLEMENTED_NETS, SUPPORTED_TASKS


def get_args() -> Namespace:
    parser = ArgumentParser(description='MathWorks Radar Drone Classification Project.')
    parser.add_argument('--task', type=str, choices=SUPPORTED_TASKS.keys(), default='binary',
                        help='indicate which task you are willing to train.')
    parser.add_argument('--arch', type=str, choices=IMPLEMENTED_NETS, default=IMPLEMENTED_NETS[0],
                        help='indicate which neural network architecture you want to use.')
    parser.add_argument('--use_pretrained', action='store_true',
                        help='indicate if use the imagenet pretrained weights.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the maximum epoch.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5,help='learning rate.')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='the minimum loss to early stop, if 0, will not early stop.')
    parser.add_argument('--ckpt_interval', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='to speed the training, only support Linux, for Windows, please give 0 instead.')
    parser.add_argument('--random_seed', type=int, default=3407,
                        help='The random seed is totally a kind of Neo-Daoism, but it do influence performance.\n'
                             'I Set it with 3407 by following this paper https://arxiv.org/abs/2109.08203.\n'
                             'The other recommended choice: 215 (which is the birthday of Taoist god,\n'
                             'as deep learning is regarded as a kind of alchemy in China)')
    parser.add_argument('--data_root', type=str, default='/home/zg34/datasets/drone_project/data',
                        help='path to the folder that contains "24GHz", "94GHz", and "207GHz" folders.')
    parser.add_argument('--train_label_path', type=str, default='/home/zg34/datasets/drone_project/train_label.csv',
                        help="path to the train_label.csv.")
    parser.add_argument('--eval_label_path', type=str, default='/home/zg34/datasets/drone_project/eval_label.csv',
                        help="path to the eval_label.csv")
    parser.add_argument('--output_dir', type=str, default='./logs',
                        help='the folder that will store the logs.')
    parser.add_argument('--log_dir', type=str,
                        help='the subfolder that in the "output_dir", '
                             'will automatic generate according to the given hyperparameters if it is not given.')
    args = parser.parse_args()
    if args.log_dir is None:
        args.log_dir = f"{args.task}({args.arch})_epoch-{args.epochs}_bs-{args.batch_size}_lr-{args.lr}"
    args.log_dir = join(args.output_dir, args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.ckpt_interval == 0:
        args.ckpt_dir = None
    else:
        args.ckpt_dir = join(args.log_dir, "checkpoints")
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(join(args.ckpt_dir, "best_models"), exist_ok=True)
    with open(join(args.log_dir, 'args.txt'), mode='w+') as f:
        print(format_args(args), file=f)
    args.num_classes = SUPPORTED_TASKS[args.task]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def format_args(args) -> str:
    args_dict = vars(args)
    return json.dumps(args_dict, indent=4)
