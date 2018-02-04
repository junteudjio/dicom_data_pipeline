'''DataParser to get numpy arrays (image, mask-1, mask-2) samples from dicoms and contours files'''
import os
import csv
import glob
import itertools
import logging
import time

import dicom
from dicom.errors import InvalidDicomError
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

__author__ = 'Junior Teudjio'
__all__ = ['DataParser']

class DataParser(object):
    class InvalidContourError(Exception):
        '''Exception raised when a contour is strictly less than 3 distinct points'''
        pass

    _plots_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    def __init__(self,  data_prefix,
                        images_dirpath,
                        masks_dirpath,
                        img_masks_filepath,
                        contours_type='i-contours.o-contours',
                        logs_prefix='_logs',
                        plots_prefix='_plots',
                        visualize_contours=True):
        '''
        Create a data_parser
        Parameters
        ----------
        data_prefix: basestring
            Parent directory where all the data for the program are saved.
        images_dirpath: basestring
            Path of the directory where to save the parsed dicom images.
        masks_dirpath: basestring
            Path of the directory where to save the parsed contours.
        img_masks_filepath: basestring
            Path of the file where to specify a pair of image,mask paths per line.
        contours_type: basestring
            Type of contour(s) to handle:
             eg:  i-contours (for single contours)
             eg: i-contours.o-contours (for multiple contours)
        logs_prefix: basestring
            Parent directory where to log files are saved.
        plots_prefix: basestring
            Parent directory where the visualization debugging plots are saved.
        visualize_contours: bool
            Decided if we should saved visualization debugging plots.
        '''
        self.images_dirpath = images_dirpath
        self.masks_dirpath = masks_dirpath
        self.img_masks_filepath = img_masks_filepath
        self.data_prefix = data_prefix
        self.contours_type = contours_type
        self.logs_prefix = logs_prefix
        self.plots_prefix = plots_prefix
        self.visualize_contours = visualize_contours

        self.dicoms_prefix = os.path.join(data_prefix, 'dicoms')
        self.contours_prefix = os.path.join(data_prefix, 'contourfiles')
        self.links_filepath = os.path.join(data_prefix, 'link.csv')

        self.parse_outfile_headers = ['img']
        self.parse_outfile_headers.extend(self.contours_type.split('.'))

        self._set_logger()


    def _set_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s:%(message)s')

        file_handler = logging.FileHandler(os.path.join(self.logs_prefix,
                                                            '{}-data-parser.log'.format(self.contours_type)))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logger

    def _log_error(self, e, *args, **kwargs):
        self.logger.error(msg='''
            error message: {}
            args: {}
            kwargs: {}
        '''.format(repr(e), str(args), str(kwargs)))

    def _get_dicoms_contours_paths(self):
        with open(self.links_filepath, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['patient_id', 'original_id'])
            next(reader)
            for row in reader:
                # first we find the images indices for which we have all contours types.
                imgs_dir = os.path.join(self.dicoms_prefix, row['patient_id'])
                imgs_filenames = os.listdir(imgs_dir)
                imgs_indices_set = set([ int(name.split('.')[0]) for name in imgs_filenames ])

                contours_dirs = []
                contours_indices_sets = []
                for contour_type in self.contours_type.split('.'):
                    contours_dir = os.path.join(self.contours_prefix, row['original_id'], contour_type)
                    contours_dirs.append(contours_dir)
                    contours_filenames = os.listdir(contours_dir)
                    contours_indices_set = set([int(name.split('-')[2]) for name in contours_filenames ])
                    contours_indices_sets.append(contours_indices_set)
                # valid_imgs_indices: list of images indices having all types of contours
                valid_imgs_indices = set.intersection(imgs_indices_set, *contours_indices_sets)
                if len(valid_imgs_indices) == 0: continue

                # yield valid images paths and their contours
                for valid_img_idx in valid_imgs_indices:
                    valid_img_path = glob.glob(os.path.join(imgs_dir, '{}.dcm'.format(valid_img_idx)))[0]
                    valid_contours_paths = []
                    contour_file_regex = '*{:04d}*'.format(valid_img_idx)
                    for contours_dir in contours_dirs:
                        valid_contour_path = glob.glob(os.path.join(contours_dir, contour_file_regex))[0]
                        valid_contours_paths.append(valid_contour_path)

                    yield valid_img_path, valid_contours_paths


    @staticmethod
    def _visualize_mask_overlay(img, masks, savepath):
        '''
        Merge the img and mask in a single figure and save to disk
        Parameters
        ----------
        img: numpy array
        masks: list of numpy array
        savepath: basestring
            path to disk where to save the merged figure.

        Returns
        -------

        '''
        plt.clf()
        fig = plt.figure()
        plt.imshow(img)
        for idx, mask in enumerate(masks):
            plt.imshow(mask, alpha=0.5)
        fig.savefig(savepath)

    @staticmethod
    def _visualize_contour_overlay(img, contours, savepath):
        '''
        Merge the img and contour points in a single figure and save to disk
        Parameters
        ----------
        img: numpy array
        contours: list of list of 2D tuples
        savepath: basestring
            path to disk where to save the merged figure.

        Returns
        -------

        '''
        plt.clf()
        fig = plt.figure()
        plt.imshow(img)
        for idx, contour in enumerate(contours):
            x = [point[0] for point in contour]
            y = [point[1] for point in contour]
            color = DataParser._plots_colors[idx % len(DataParser._plots_colors)]
            plt.plot(x, y, alpha=1, color=color)
        fig.savefig(savepath)

    @staticmethod
    def _parse_contour_file(filepath):
        '''
        Parse the given contour filepath and return the list of x,y coordinates of the contour.
        Parameters
        ----------
        filepath: basestring
            Path to the contour file

        Returns
        -------
            list of tuples holding x, y coordinates of the contour
        '''
        coords_lst = []
        with open(filepath, 'r') as infile:
            for line in infile:
                coords = line.strip().split()

                x_coord = float(coords[0])
                y_coord = float(coords[1])
                coords_lst.append((x_coord, y_coord))

        if len(coords_lst) < 3:
            raise DataParser.InvalidContourError('''Invalid contour file:{}
             cannot create a contour with less than 3 points: only {} points'''.format(filepath, len(coords_lst)))
        return coords_lst

    @staticmethod
    def _parse_dicom_file(filepath):

        '''
        Parse a dicom file and return eventually the image data as numpy array
        Parameters
        ----------
        filepath: basestring
            Path to the dicom file

        Returns
        -------
        numpy array

        '''
        try:
            dcm = dicom.read_file(filepath)
            dcm_image = dcm.pixel_array

            try:
                intercept = dcm.RescaleIntercept
            except AttributeError:
                intercept = 0.0
            try:
                slope = dcm.RescaleSlope
            except AttributeError:
                slope = 0.0

            if intercept != 0.0 and slope != 0.0:
                dcm_image = dcm_image * slope + intercept
            return dcm_image
        except InvalidDicomError as e:
            raise e

    @staticmethod
    def _poly_to_mask(polygon, width, height):
        '''
        Convert a polygon to a mask
        Parameters
        ----------
        polygon: list
            list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
        width: scalar
            image width
        height: scalar
            image height

        Returns
        -------
            A boolean numpy array of shape(width, height)
        '''

        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
        mask = np.array(img).astype(bool)
        return mask

    def parse(self):
        '''
        Method to call to parse the dicoms and contours files.
        Returns
        -------

        '''
        start = time.time()
        self.logger.info('Parsing dicom and contours files started ...')

        outfile = open(self.img_masks_filepath, 'w')
        # write header line
        header_line = ' '.join(self.parse_outfile_headers)
        outfile.write('{}\n'.format(header_line))

        dcm_contours_paths = list(self._get_dicoms_contours_paths())
        for idx, (dcm_path, contour_paths) in enumerate(dcm_contours_paths):
            try:
                img = DataParser._parse_dicom_file(dcm_path)
                contours, masks = [], []
                for contour_path in contour_paths:
                    contour = DataParser._parse_contour_file(contour_path)
                    mask = DataParser._poly_to_mask(contour, img.shape[0], img.shape[1])
                    contours.append(contour)
                    masks.append(mask)
            except InvalidDicomError:
                self.logger.error('Error loading dicom file: {}'.format(dcm_path), exc_info=True)
            except DataParser.InvalidContourError:
                self.logger.error('Error loading contour file: {}'.format(contour_path), exc_info=True)
            except Exception:
                self.logger.error('Something went wrong: ', exc_info=True)
            else:
                # list accumulator for img-path and mask-1-path, mask-2-path, etc...
                sample_line = []

                # save the img to the disk
                img_path = os.path.join(self.images_dirpath, '{}.npy'.format(idx))
                sample_line.append(img_path)
                np.save(img_path, img)

                for contour_type, mask in itertools.izip(self.contours_type.split('.'), masks):
                    mask_path = os.path.join(self.masks_dirpath, '{}.{}.npy'.format(idx, contour_type))
                    sample_line.append(mask_path)
                    np.save(mask_path, mask)

                # save their paths  to file
                sample_line = ' '.join(sample_line)
                outfile.write('{}\n'.format(sample_line))

                # for visual debugging purposes also create an image for each of the first 10
                # where the image and mask are merged together to see if everything is ok.
                if self.visualize_contours and idx < 5:
                    mask_overlay_path = os.path.join(self.plots_prefix,
                                                     '{}.mask_overlay.{}.jpg'.format(self.contours_type, idx))
                    contour_overlay_path = os.path.join(self.plots_prefix,
                                                        '{}.contour_overlay.{}.jpg'.format(self.contours_type, idx))
                    DataParser._visualize_mask_overlay(img, masks, mask_overlay_path)
                    DataParser._visualize_contour_overlay(img, contours, contour_overlay_path)

        outfile.close()

        end = time.time()
        self.logger.info('Parsing ended successfully after: {} seconds'.format(end-start))
        self.logger.debug('''Please refer to the followings output file and folders:
            parsed numpy images: {imgs_path}
            parsed numpy masks: {masks_path}
            images-contours  file: {img_masks_filepath}
            visualizing contours and masks figures: {plots_prefix}/*.jpg
        '''.format(
            imgs_path=self.images_dirpath,
            masks_path=self.masks_dirpath,
            img_masks_filepath=self.img_masks_filepath,
            plots_prefix=self.plots_prefix))