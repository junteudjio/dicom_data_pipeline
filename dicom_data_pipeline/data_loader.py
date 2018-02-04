'''DataLoader based on Pytorch very convenient utility classes:
        http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
import linecache

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

__author__ = 'Junior Teudjio'
__all__ = ['DicomMasksDataset', 'DataLoaderFactory', 'to_tensor', 'to_ndarray']


def to_tensor(sample):
    '''
    Transform a ndarray array sample into a Tensorflow sample
    Parameters
    ----------
    sample: dict ('img':np.int16, 'i-contour':np.bool, 'o-contour':np.bool, ...)

    Returns
    -------
    sample: dict ('img':Tensorflow.ShortTensor,
                  'i-contour':Tensorflow.ByteTensor,
                  'o-contour':Tensorflow.ByteTensor,
                  ...)
    '''
    new_sample = dict()
    for key, val in sample.iteritems():
        if key == 'img':
            new_sample[key] = torch.from_numpy(val)
        else: # handle contour mask differently
            new_sample[key] = torch.from_numpy(val.astype(np.uint8))

    return new_sample

def to_ndarray(sample):
    '''
    Transform a Tensorflow array sample into a ndarray sample
    Parameters
    ----------
    sample: dict ('img':Tensorflow.ShortTensor,
                  'i-contour':Tensorflow.ByteTensor,
                  'o-contour':Tensorflow.ByteTensor,
                  ...)

    Returns
    -------
    sample: dict ('img':np.int16, 'i-contour':np.bool, 'o-contour':np.bool, ...)
    '''
    new_sample = dict()
    for key, val in sample.iteritems():
        if key == 'img':
            new_sample[key] = val.numpy()
        else:  # handle contour mask differently
            new_sample[key] = val.numpy().astype(np.bool)

    return new_sample

class DicomMasksDataset(Dataset):
    ''' Dicom & mask images dataset.'''

    def __init__(self, img_masks_filepath, transform=None):
        '''
        Create a dataset object to iterate over a file containing in each line a path to:
        img, contour-1-mask, contour-2-mask, etc... numpy arrays.
        Parameters
        ----------
        img_masks_filepath: basestring
            Path of the file where are saved image,masks paths per line.
        transform: callable or None
            A transformation to apply to a training sample.
            Each sample being: a dict(image, contour-1-mask, contour-2-mask).
            An example of  function can be a conversion from numpy array to Pytorch/Keras/Tensorflow/... tensors.
        '''

        samples_count = 0
        with open(img_masks_filepath, 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                # first line for the header:
                if idx == 0:
                    self.headers = line.split(' ')
                    continue
                # next lines contains paths to img, and contours-masks paths
                if line:
                    samples_count += 1

        self.samples_count = samples_count
        self.img_masks_filepath = img_masks_filepath
        self.transform = transform

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        if idx >= self.samples_count:
            raise StopIteration

        idx += 2 #add 2 since linecache indexing starts at 1 not 0, and line 1 is for header

        paths = linecache.getline(self.img_masks_filepath, idx).strip().split(' ')
        sample_paths = dict(zip(self.headers, paths))
        sample = dict((key, np.load(path)) for key, path in sample_paths.iteritems())
        if self.transform:
            sample = self.transform(sample)

        return sample


class DataLoaderFactory(object):
    @staticmethod
    def get_data_loader(img_masks_filepath, tensor_or_ndarray='ndarray', batch_size=8, shuffle=True, num_workers=1):
        '''
        Encasulate the creation of the Pytorch Dataset object and then the DataLoader object given the parameters.
        Parameters
        ----------
        img_masks_filepath: basestring
            Path of the file where are saved a pair of image,mask paths per line.
        tensor_or_ndarray: basestring
           Decides to return a torch tensor or a numpy ndarray: values to choose between ndarray and tensor
        batch_size: int
            The number of samples per batch
        shuffle: bool
            Load the samples randomly from the entire dataset.
        num_workers: int
            If multi-core/GPUs use this parameter to load data in parallel.

        Returns
        -------
        data_loader: DataLoader
             which generates either Torch Tensors or numpy ndarrays per batch
             depending of the value of tensor_or_ndarray.
        '''

        # first instantiate the dataset object
        dicom_mask_dataset = DicomMasksDataset(img_masks_filepath=img_masks_filepath, transform=to_tensor)

        # now create the data_loader object
        data_loader = DataLoader(
            dataset=dicom_mask_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        if tensor_or_ndarray == 'ndarray':
            data_loader = (to_ndarray(sample) for sample in data_loader)

        return data_loader