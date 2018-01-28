'''DataParser to get numpy arrays (image, mask) pairs from dicoms and contours files'''
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

    def __init__(self,  data_prefix,
                        images_dirpath,
                        masks_dirpath,
                        image_mask_pairs_filepath,
                        contours_type='i-contours',
                        logs_prefix='_logs',
                        plots_prefix='_plots',
                        visualize_contours=True):

        self.images_dirpath = images_dirpath
        self.masks_dirpath = masks_dirpath
        self.image_mask_pairs_filepath = image_mask_pairs_filepath
        self.data_prefix = data_prefix
        self.contours_type = contours_type
        self.logs_prefix = logs_prefix
        self.plots_prefix = plots_prefix
        self.visualize_contours = visualize_contours

        self.dicoms_prefix = os.path.join(data_prefix, 'dicoms')
        self.contours_prefix = os.path.join(data_prefix, 'contourfiles')
        self.links_filepath = os.path.join(data_prefix, 'link.csv')

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

    def _get_dicoms_contours_pairs_paths(self):
        with open(self.links_filepath, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=['patient_id', 'original_id'])
            next(reader)
            for row in reader:
                imgs_dir = os.path.join(self.dicoms_prefix, row['patient_id'])
                contours_dir = os.path.join(self.contours_prefix, row['original_id'], self.contours_type)

                contours_paths = glob.glob(os.path.join(contours_dir, '*-manual.txt'))
                contour_filenames = (path.split('/')[-1] for path in contours_paths)
                contours_indices = (int(name.split('-')[2]) for name in contour_filenames)

                for contour_idx, contour_path in itertools.izip(contours_indices, contours_paths):
                    # check if there is a matching dicom file for this contour index
                    imgs_paths = glob.glob(os.path.join(imgs_dir, '{}.dcm'.format(contour_idx)))

                    if len(imgs_paths) == 1:
                        yield imgs_paths[0], contour_path
                    else:
                        self.logger.warning(
                            msg='No mathing dicom file for contour file: {}'.format(contour_path)
                        )

    def _visualize_mask_overlay(self, img, mask, savepath):
        '''
        Merge the img and mask in a single figure and save to disk
        Parameters
        ----------
        img: numpy array
        mask: numpy array
        savepath: basestring
            path to disk where to save the merged figure.

        Returns
        -------

        '''
        plt.clf()
        fig = plt.figure()
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        fig.savefig(savepath)

    def _visualize_contour_overlay(self, img, contour, savepath):
        '''
        Merge the img and contour points in a single figure and save to disk
        Parameters
        ----------
        img: numpy array
        contour: list of 2D tuples
        savepath: basestring
            path to disk where to save the merged figure.

        Returns
        -------

        '''
        plt.clf()
        fig = plt.figure()
        plt.imshow(img)
        x = [point[0] for point in contour]
        y = [point[1] for point in contour]
        plt.plot(x, y, alpha=1, color='r')
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
            raise DataParser.InvalidContourError('Invalid contour file:{} , less than 3 contours: {}'.format(
                filepath, len(coords_lst)
            ))
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
        numpy array or None

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

        outfile = open(self.image_mask_pairs_filepath, 'w')
        dcms_contours_paths = self._get_dicoms_contours_pairs_paths()
        for idx, (dcm_path, contour_path) in enumerate(dcms_contours_paths):
            try:
                img = DataParser._parse_dicom_file(dcm_path)
                contour = DataParser._parse_contour_file(contour_path)
                mask = DataParser._poly_to_mask(contour, img.shape[0], img.shape[1])
            except InvalidDicomError as e:
                self._log_error(e, DataParser._parse_dicom_file.__name__, dcm_path=dcm_path)
            except DataParser.InvalidContourError as e:
                self._log_error(e, DataParser._parse_contour_file.__name__, contour_path=contour_path)
            except Exception as e:
                self._log_error(e, dcm_path=dcm_path, contour_path=contour_path)
            else:
                # save the img to the disk
                img_path = os.path.join(self.images_dirpath, '{}.npy'.format(idx))
                mask_path = os.path.join(self.masks_dirpath, '{}.npy'.format(idx))
                np.save(img_path, img)
                np.save(mask_path, mask)

                # save their paths pairs to file
                outfile.write('{img_path} {mask_path}\n'.format(img_path=img_path, mask_path=mask_path))

                # for visual debugging purposes also create an image for each of the first 10 pairs
                # where the image and mask are merged together to see if everything is ok.
                if self.visualize_contours and idx < 5:
                    mask_overlay_path = os.path.join(self.plots_prefix,
                                                     '{}-mask-overlay-{}.jpg'.format(self.contours_type, idx))
                    contour_overlay_path = os.path.join(self.plots_prefix,
                                                        '{}-overlay-{}.jpg'.format(self.contours_type, idx))
                    self._visualize_mask_overlay(img, mask, mask_overlay_path)
                    self._visualize_contour_overlay(img, contour, contour_overlay_path)

        outfile.close()

        end = time.time()
        self.logger.info('Parsing ended successfully after: {} seconds'.format(end-start))
        self.logger.debug('''Please refer to the followings output file and folders:
            parsed numpy images: {imgs_path}
            parsed numpy masks: {masks_path}
            images-contours pairs file: {img_contour_pairs_file}
            visualizing contours and masks figures: {plots_prefix}/*.jpg
        '''.format(
            imgs_path=self.images_dirpath,
            masks_path=self.masks_dirpath,
            img_contour_pairs_file=self.image_mask_pairs_filepath,
            plots_prefix=self.plots_prefix))