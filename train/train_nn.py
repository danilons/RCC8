# coding: utf-8
import argparse
from dataset import pascal_voc as voc_dataset


def parse_args():
    """
    Parse input

    :return: obj of parsed args
    """
    parser = argparse.ArgumentParser(description='Train an one layer NN network')
    parser.add_argument('--dset', help='Dataset (default PASCAL VOC 2007)')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()