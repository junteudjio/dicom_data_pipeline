import unittest
import numpy as np

from dicom_data_pipeline.data_parser import DataParser

__author__ = 'Junior Teudjio'


class TestDataParser(unittest.TestCase):

    def test_parse_contour_file_1(self):
        '''Test the loading of a correct contour file'''
        contours = DataParser._parse_contour_file('../tests/data/correct-contour.txt')
        self.assertTrue(isinstance(contours, list))
        self.assertTrue(len(contours)>=3)
        self.assertTrue(isinstance(contours[0], tuple))
        self.assertTrue(isinstance(contours[0][0], float))

    def test_parse_contour_file_2(self):
        '''Test the loading of an ibcorrect contour file'''
        self.assertRaises(DataParser.InvalidContourError,
                          DataParser._parse_contour_file,
                           '../tests/data/incorrect-contour.txt')



    def test_parse_dicom_file_1(self):
        '''Test the loading of a correct dicom file'''
        img = DataParser._parse_dicom_file('../tests/data/correct.dcm')
        self.assertTrue(isinstance(img, np.ndarray))

    def test_parse_dicom_file_2(self):
        '''Test the loading of an incorrect dicom file'''
        self.assertRaises(Exception, DataParser._parse_dicom_file, '../tests/data/incorrect.dcm')


    def test_poly_to_mask(self):
        '''Test if the conversion from polygon to mask is correct'''
        contours = DataParser._parse_contour_file('../tests/data/correct-contour.txt')
        img = DataParser._parse_dicom_file('../tests/data/correct.dcm')

        mask = DataParser._poly_to_mask(contours,img.shape[0], img.shape[1])

        self.assertEqual(mask.shape, img.shape)
        self.assertEqual(mask.dtype, np.bool)

if __name__ == '__main__':
    unittest.main()