'''DataLoader based on Pytorch very convenient utility classes:
        http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
import linecache

import numpy as np
from torch.utils.data import Dataset, DataLoader

__author__ = 'Junior Teudjio'

class DicomMaskPairsDataset(Dataset):
    ''' Dicom & mask images dataset.'''

    def __init__(self, pairs_filepath, transform=None):
        '''
        Create a dataset object to iterate over a file containing in each line a path to 2 numpy arrays.
        Parameters
        ----------
        pairs_filepath: basestring
            Path of the file where are saved a pair of image,mask paths per line.
        transform: list of functions or None
            A sequence of transformations to apply to a training sample.
            Each sample being: a dict(image, mask).
            An example of  function can be a conversion from numpy array to Pytorch/Keras/Tensorflow/... tensors.
        '''

        samples_count = 0
        with open(pairs_filepath, 'rb') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples_count += 1

        self.samples_count = samples_count
        self.pairs_filepath = pairs_filepath
        self.transform = transform

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        idx += 1 #add 1 since linecache indexing starts at 1 not 0
        paths = linecache.getline(self.pairs_filepath, idx).strip()
        img_path, mask_path = paths.split(' ')

        img = np.load(img_path)
        mask = np.load(mask_path)

        sample = {'img': img, 'mask':mask}
        if self.transform:
            sample = self.transform(sample)

        return sample


class DataLoaderFactory(object):
    @staticmethod
    def get_data_loader(pairs_filepath, transform=None, batch_size=8, shuffle=True, num_workers=1):
        '''
        Encasulate the creation of the Pytorch Dataset object and then the DataLoader object given the parameters.
        Parameters
        ----------
        pairs_filepath: basestring
            Path of the file where are saved a pair of image,mask paths per line.
        transform: list of functions or None
            A sequence of transformations to apply to a training sample.
            Each sample being: a dict(image, mask).
            An example of function can be a conversion from numpy array to Pytorch/Keras/Tensorflow/... tensors.
        batch_size: int
            The number of samples per batch
        shuffle: bool
            Load the samples randomly from the entire dataset.
        num_workers: int
            If multi-core/GPUs use this parameter to load data in parallel.

        Returns
        -------
        data_loader: DataLoader
        '''

        # first instantiate the dataset object
        dicom_mask_dataset = DicomMaskPairsDataset(pairs_filepath=pairs_filepath, transform=transform)

        # now create the data_loader object
        data_loader = DataLoader(
            dataset=dicom_mask_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return data_loader
