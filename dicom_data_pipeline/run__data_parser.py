import argparse
import os

import utils
from data_parser import DataParser

__author__ = 'Junior Teudjio'


def _setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-prefix', type=str, default='_data', help='path containing all the program\'s data')
    parser.add_argument('--logs-prefix', type=str, default='_logs', help='path containing all the program\'s logs')
    parser.add_argument('--plots-prefix', type=str, default='_plots', help='path containing all the program\'s plots')

    parser.add_argument('--contours-type', type=str, default='i-contours', help='The contours type to process')
    parser.add_argument('--no-visualize-contours', default=False, action='store_true',
                            help='Create few images to visualize if contours are loaded correctly or not')

    parser.add_argument('--masks-dir-prefix', type=str, default='masks',
                            help='The prefix of directory where to save extracted numpy masks from contours')
    parser.add_argument('--images-dir-prefix', type=str, default='images',
                        help='The prefix of directory where to save extracted numpy images from dicoms')
    parser.add_argument('--image-mask-pairs-file-prefix', type=str, default='image-mask-pairs',
                            help='The prefix of the file where the (img_path, mask_path) couples will be saved')

    return parser.parse_args()


def _extend_args(args):
    assert args.contours_type in ['i-contours', 'o-contours']
    masks_dirpath = os.path.join(args.data_prefix, '-'.join([args.masks_dir_prefix, args.contours_type]))
    images_dirpath = os.path.join(args.data_prefix, '-'.join([args.images_dir_prefix, args.contours_type]))
    image_mask_pairs_filepath = os.path.join(args.data_prefix, '-'.join([args.image_mask_pairs_file_prefix,
                                                                           args.contours_type]))
    image_mask_pairs_filepath += '.csv'
    # This is the directory containing the numpy array masks extracted from contours
    args.masks_dirpath = masks_dirpath

    # This is the directory containing the numpy array images extracted from dicoms
    args.images_dirpath = images_dirpath

    # This is the file containing in each line a couple :  dcm_image_path, mask_path needed by the data loader.
    args.image_mask_pairs_filepath = image_mask_pairs_filepath

    args.visualize_contours = not args.no_visualize_contours


def main(args):
    data_parser = DataParser(data_prefix=args.data_prefix,
                             images_dirpath=args.images_dirpath,
                             masks_dirpath=args.masks_dirpath,
                             image_mask_pairs_filepath=args.image_mask_pairs_filepath,
                             contours_type=args.contours_type,
                             logs_prefix=args.logs_prefix,
                             visualize_contours=args.visualize_contours)
    data_parser.parse()


if __name__ == '__main__':
    # setup program context.
    args = _setup_args()
    _extend_args(args)
    utils.mkdir_p(args.logs_prefix)
    utils.mkdir_p(args.plots_prefix)
    utils.mkdir_p(args.masks_dirpath)
    utils.mkdir_p(args.images_dirpath)

    # run program
    main(args)