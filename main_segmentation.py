import os
import sys
import shutil
import time
import numpy as np
from shutil import copyfile
from tqdm import tqdm

from utils import vararg_callback_bool, vararg_callback_int
from dataloader import  *
import argparse

import torch
from engine import segmentation_engine
from utils import torch_dice_coef_loss

sys.setrecursionlimit(40000)


def get_args_parser():
    parser = argparse.ArgumentParser(description='Command line arguments for segmentation target tasks.')
    parser.add_argument('--data_dir', help='dataset directory', default=None)
    parser.add_argument("--train_list", help="file for training list", default=None)
    parser.add_argument("--val_list",  help="file for validating list", default=None)
    parser.add_argument("--test_list",  help="file for test list", default=None)
    parser.add_argument('--batch_size',  default=32, type=int)
    parser.add_argument("--data_set", help="Montgomery|JSRTLung|JSRTClavicle|JSRTHeart", default="Montgomery")
    parser.add_argument("--pretrained_weights", help="Path to the Pretrained model", default=None)
    parser.add_argument("--key", help="key name in the pretrained checkpoint", default="state_dict")
    parser.add_argument('--epochs', help='number of epochs', default=200, type=int)
    parser.add_argument("--num_class", help="number of the classes in the downstream task", default=1, type=int)
    parser.add_argument('--train_num_workers', help='train num of parallel workers for data loader', default=2,
                        type=int)
    parser.add_argument('--test_num_workers', help='test num of parallel workers for data loader', default=2, type=int)
    parser.add_argument('--distributed', help='whether to use distributed or not', dest='distributed',
                        action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--learning_rate', help='learning rate', default=0.001, type=float)
    parser.add_argument('--mode', help='train|test', default='train')

    parser.add_argument('--arch', help='segmentation network architecture', default='upernet_swin')
    parser.add_argument('--proxy_dir', help='path to pre-trained model', default=None)
    parser.add_argument('--device', help='cuda|cpu', default="cuda")
    parser.add_argument('--num_trial', default=1, help='trial number', type=int)
    parser.add_argument("--start_index", help="the start model index", default=0, type=int)
    parser.add_argument('--init', help='Random |ImageNet |or other pre-trained methods', default='Random')
    parser.add_argument('--normalization', help='imagenet|None', default='imagenet')
    parser.add_argument('--activate', help='activation', default="sigmoid")
    parser.add_argument('--patience', type=int, default=20, help='num of patient epochs')
    parser.add_argument('--few_shot', help='number or percentage of training samples', default=0, type=float)

    args = parser.parse_args()
    return args

def main(args):
    print(args)
    assert args.data_dir is not None
    assert args.train_list is not None
    assert args.val_list is not None
    assert args.test_list is not None

    model_path = os.path.join("./Models/Segmentation", args.data_set, args.arch, args.init)
    
    if args.data_set == "Montgomery":
        dataset_train = Montgomery(args.data_dir,args.train_list,augment=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = Montgomery(args.data_dir,args.val_list,augment=build_transform_segmentation(), normalization=args.normalization)
        dataset_test = Montgomery(args.data_dir,args.test_list,augment=None, normalization=args.normalization)
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "JSRTLung":
        dataset_train = JSRTLung(args.data_dir,args.train_list,augment=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = JSRTLung(args.data_dir,args.val_list,augment=None, normalization=args.normalization)
        dataset_test = JSRTLung(args.data_dir,args.test_list,augment=None, normalization=args.normalization)
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "JSRTClavicle":
        dataset_train = JSRTClavicle(args.data_dir,args.train_list,augment=build_transform_segmentation(), few_shot=args.few_shot, normalization=args.normalization)
        dataset_val = JSRTClavicle(args.data_dir,args.val_list,augment=build_transform_segmentation(), few_shot=args.few_shot, normalization=args.normalization)
        dataset_test = JSRTClavicle(args.data_dir,args.test_list,augment=None, few_shot=args.few_shot, normalization=args.normalization)
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test)

    elif args.data_set == "JSRTHeart":
        dataset_train = JSRTHeart(args.data_dir,args.train_list,augment=build_transform_segmentation(), normalization=args.normalization)
        dataset_val = JSRTHeart(args.data_dir,args.val_list,augment=None, normalization=args.normalization)
        dataset_test = JSRTHeart(args.data_dir,args.test_list,augment=None, normalization=args.normalization)
        segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test)

if __name__ == '__main__':
    args = get_args_parser()
    main(args)