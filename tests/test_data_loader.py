import unittest

import numpy as np
import torch

from dicom_data_pipeline.data_loader import DicomMasksDataset, DataLoaderFactory, to_tensor
import matplotlib.pyplot as plt


__author__ = 'Junior Teudjio'

SHOW_DEBUG_PLOTS = False
SAVE_DEBUG_PLOTS = True

class TestDicomMasksDataset(unittest.TestCase):

    def test_without_transform(self):
        dicom_mask_dataset = DicomMasksDataset(img_masks_filepath='_data/image-masks.i-contours.csv')
        fig = plt.figure()
        max_iter = 5
        for i in xrange(len(dicom_mask_dataset)):
            sample = dicom_mask_dataset[i]
            self.assertEqual(sample['img'].shape, sample['i-contours'].shape)
            self.assertEqual(sample['img'].dtype, np.int16)
            self.assertEqual(sample['i-contours'].dtype, np.bool)

            ax = plt.subplot(1, max_iter, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            ax.imshow(sample['img'])
            ax.imshow(sample['i-contours'], alpha=0.5)
            if i == max_iter-1:
                if SAVE_DEBUG_PLOTS:
                    fig.savefig('../tests/data/test-data-loader-1.jpg')
                if SHOW_DEBUG_PLOTS:
                    plt.show()
                break

    def test_with_transform(self):
        dicom_mask_dataset = DicomMasksDataset(img_masks_filepath='_data/image-masks.i-contours.csv',
                                                   transform=to_tensor)
        for i in range(len(dicom_mask_dataset)):
            sample = dicom_mask_dataset[i]
            self.assertTrue(isinstance(sample['img'], torch.ShortTensor))
            self.assertTrue(isinstance(sample['i-contours'], torch.ByteTensor))


    def test_multi_contours_parse(self):
        dicom_mask_dataset = DicomMasksDataset(img_masks_filepath='_data/image-masks.i-contours.o-contours.csv')
        fig = plt.figure()
        max_iter = 5
        for i in xrange(len(dicom_mask_dataset)): # o-contours
            sample = dicom_mask_dataset[i]
            self.assertEqual(sample['img'].shape, sample['i-contours'].shape)
            self.assertEqual(sample['img'].shape, sample['o-contours'].shape)
            self.assertEqual(sample['img'].dtype, np.int16)
            self.assertEqual(sample['i-contours'].dtype, np.bool)
            self.assertEqual(sample['o-contours'].dtype, np.bool)

            ax = plt.subplot(1, max_iter, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            ax.imshow(sample['img'])
            ax.imshow(sample['i-contours'], alpha=0.5)
            ax.imshow(sample['o-contours'], alpha=0.5)
            if i == max_iter-1:
                if SAVE_DEBUG_PLOTS:
                    fig.savefig('../tests/data/test-data-loader-multi-contours-1.jpg')
                if SHOW_DEBUG_PLOTS:
                    plt.show()
                break


class TestDataLoader(unittest.TestCase):

    def test_data_loader__tensor_batch(self):
        '''Test the dataloader as a Pytorch tensor generator.'''
        max_iter = 2

        batch_size = 8
        data_loader = DataLoaderFactory.get_data_loader('_data/image-masks.i-contours.csv',
                                                        tensor_or_ndarray='tensor', # choose generation type.
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=1)

        for batch_idx, batch_samples in enumerate(data_loader):
            imgs_batch = batch_samples['img']
            masks_batch = batch_samples['i-contours']

            self.assertEqual(len(imgs_batch), batch_size)
            self.assertEqual(len(masks_batch), batch_size)
            self.assertTrue(isinstance(imgs_batch[0], torch.ShortTensor))
            self.assertTrue(isinstance(masks_batch[0], torch.ByteTensor))

            if batch_idx == max_iter:
                break

    def test_data_loader__ndarray_batch(self):
        '''Test the dataloader as a numpy ndarray generator.'''
        max_iter = 2

        batch_size = 2
        data_loader = DataLoaderFactory.get_data_loader('_data/image-masks.i-contours.csv',
                                                        tensor_or_ndarray='ndarray',  # choose generation type.
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=1)

        for batch_idx, batch_samples in enumerate(data_loader):
            imgs_batch = batch_samples['img']
            masks_batch = batch_samples['i-contours']

            self.assertEqual(len(imgs_batch), batch_size)
            self.assertEqual(len(masks_batch), batch_size)
            self.assertTrue(isinstance(imgs_batch[0], np.ndarray))
            self.assertTrue(isinstance(masks_batch[0], np.ndarray))
            self.assertEqual(imgs_batch[0].dtype, np.int16)
            self.assertEqual(masks_batch[0].dtype, np.bool)

            if batch_idx == max_iter:
                break

        # plot imgs and masks for the last batch
        if SAVE_DEBUG_PLOTS or SHOW_DEBUG_PLOTS:
            plt.clf()
            fig = plt.figure()
            for i in xrange(batch_size):
                ax = plt.subplot(1, batch_size, i + 1)
                plt.tight_layout()
                ax.set_title('Sample #{}'.format(i, batch_idx))
                ax.axis('off')
                ax.imshow(imgs_batch[i])
                ax.imshow(masks_batch[i], alpha=0.5)
                if i == batch_size - 1:
                    if SAVE_DEBUG_PLOTS:
                        fig.savefig('../tests/data/test-data-loader-batch-{}.jpg'.format(batch_idx))
                    if SHOW_DEBUG_PLOTS:
                        plt.show()
                    break

    def test_data_loader_multi_contours__ndarray_batch(self):
        '''Test the multi contours dataloader as a numpy ndarray generator.'''
        max_iter = 2

        batch_size = 2
        data_loader = DataLoaderFactory.get_data_loader('_data/image-masks.i-contours.o-contours.csv',
                                                        tensor_or_ndarray='ndarray',  # choose generation type.
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=1)

        for batch_idx, batch_samples in enumerate(data_loader):
            imgs_batch = batch_samples['img']
            i_masks_batch = batch_samples['i-contours']
            o_masks_batch = batch_samples['o-contours']

            self.assertEqual(len(imgs_batch), batch_size)
            self.assertEqual(len(i_masks_batch), batch_size)
            self.assertEqual(len(o_masks_batch), batch_size)
            self.assertTrue(isinstance(imgs_batch[0], np.ndarray))
            self.assertTrue(isinstance(i_masks_batch[0], np.ndarray))
            self.assertTrue(isinstance(o_masks_batch[0], np.ndarray))
            self.assertEqual(imgs_batch[0].dtype, np.int16)
            self.assertEqual(i_masks_batch[0].dtype, np.bool)
            self.assertEqual(o_masks_batch[0].dtype, np.bool)

            if batch_idx == max_iter:
                break

        # plot imgs and masks for the last batch
        if SAVE_DEBUG_PLOTS or SHOW_DEBUG_PLOTS:
            plt.clf()
            fig = plt.figure()
            for i in xrange(batch_size):
                ax = plt.subplot(1, batch_size, i + 1)
                plt.tight_layout()
                ax.set_title('Sample #{}'.format(i, batch_idx))
                ax.axis('off')
                ax.imshow(imgs_batch[i])
                ax.imshow(i_masks_batch[i], alpha=0.5)
                ax.imshow(o_masks_batch[i], alpha=0.5)
                if i == batch_size - 1:
                    if SAVE_DEBUG_PLOTS:
                        fig.savefig('../tests/data/test-data-loader-multi-contours-batch-{}.jpg'.format(batch_idx))
                    if SHOW_DEBUG_PLOTS:
                        plt.show()
                    break


if __name__ == '__main__':
    unittest.main()