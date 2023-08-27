# -*- coding: utf-8 -*-
import os
import numpy as np
from osgeo import gdal
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    """
    Custom data generator class for loading and processing image data with labels.
    Inherits from keras.utils.Sequence for efficient data loading and shuffling.

    Parameters
    ----------
    img_dir : str
        Directory containing image data.
    lbl_dir : str
        Directory containing label data.
    list_ids : list
        List of IDs corresponding to data samples.
    batch_size : int, optional
        Batch size for data generation. Default is 1.
    img_size : tuple, optional
        Size of input images. Default is (256, 256).
    n_channels : int, optional
        Number of image channels. Default is 3.
    n_classes : int, optional
        Number of classes/labels. Default is 1.
    shuffle : bool, optional
        Shuffle data at the end of each epoch. Default is True.
    lbl_scale : list, optional
        List of scaling factors for labels. Default is [255].
    weight_layer : bool, optional
        Include weight layer in output. Default is True.
    """
    
    def __init__(self, img_dir, lbl_dir, list_ids, batch_size=1, img_size=(256, 256), 
                 n_channels=3, n_classes=1, shuffle=True, 
                 lbl_scale=[255], weight_layer=True):
        self.img_dir = img_dir
        self.list_ids = list_ids
        self.lbl_dir = lbl_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.lbl_scale = lbl_scale
        self.weight_layer = weight_layer
        self.on_epoch_end()


    def __len__(self):
        """
        Return the number of batches per epoch.

        Returns
        -------
        int
            Number of batches per epoch.
        """
        
        return int(np.floor(len(self.list_ids) / self.batch_size))


    def __getitem__(self, index):
        """
        Generate and return a batch of data.

        Parameters
        ----------
        index : int
            Index of the batch.

        Returns
        -------
        tuple
            If weight_layer is True, returns (X, y, z). Otherwise, returns (X, y).
        """
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        list_ids_temp = [self.list_ids[k] for k in indexes]
        
        if self.weight_layer:
            X, y, z = self.__data_generation(list_ids_temp)
            return X, y, z
        else:
            X, y = self.__data_generation(list_ids_temp)
            return X, y


    def on_epoch_end(self):
        """
        Shuffle the data indexes at the end of each epoch.
        """
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        
    def __scale_lbl(self, lbl_array):
        """
        Scale the label array using the provided scaling factors.

        Parameters
        ----------
        lbl_array : numpy.ndarray
            Label array to be scaled.

        Returns
        -------
        numpy.ndarray
            Scaled label array.
        """
        lbl_array = lbl_array.astype(np.float32) / self.lbl_scale
        
        return lbl_array
        
    
    def __data_generation(self, list_ids_temp):
        """
        Generate a batch of data.

        Parameters
        ----------
        list_ids_temp : list
            List of IDs corresponding to the current batch.

        Returns
        -------
        tuple
            If weight_layer is True, returns (X, y, z). Otherwise, returns (X, y).
        """
        X = np.empty((self.batch_size, *self.img_size, self.n_channels))
        y = np.empty((self.batch_size, *self.img_size, self.n_classes))

        if self.weight_layer:
            z = np.empty((self.batch_size, *self.img_size, 1))

        for i, ID in enumerate(list_ids_temp):
            # load the image and label
            img_ds = gdal.Open(os.path.join(self.img_dir, f"{ID}.tif"))
            img = img_ds.ReadAsArray()
            img = np.transpose(img, (1, 2, 0)) # Swap channels last
            
            label_ds = gdal.Open(os.path.join(self.lbl_dir, f"{ID}.tif"))
            label = label_ds.ReadAsArray()
            
            if self.weight_layer:
                weight = np.expand_dims(label[1], axis=-1)
                z[i,] = weight
                
            label = self.__scale_lbl(label[0]).astype(np.float32)
            
            if self.n_classes == 1:
                label = np.expand_dims(label, axis=-1)
            
            # store the image and label
            X[i,] = img.astype(np.float32)
            y[i,] = label
            
        X[X > 1.0] = 1.0

        if self.weight_layer:
            return X, y, z
        else:
            return X, y