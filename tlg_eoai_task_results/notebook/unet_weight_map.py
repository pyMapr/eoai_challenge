# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:22:45 2023

@author: Chuck1
"""


import sys
import numpy as np
import cv2


class UnetWeightMap:
    def __init__(self, mask, wc={0: 1, 1: 5}, w0=10, sigma=5): # {0: 1, # background, 1: 5  # objects}
        """
        Generate weight maps as specified in the U-Net paper
        for boolean mask.
        
        "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        https://arxiv.org/pdf/1505.04597.pdf
        
        Parameters
        ----------
        mask: Numpy array
            2D array of shape (image_height, image_width) representing binary mask
            of objects.
        wc: dict
            Dictionary of weight classes.
        w0: int
            Border weight parameter.
        sigma: int
            Border width parameter.
        Returns
        -------
        Numpy array
            Training weights. A 2D array of shape (image_height, image_width).
        """
        self.mask = mask
        self.wc = wc
        self.w0 = w0
        self.sigma = sigma
        
    def weight_map(self):
        # Compute connected components using OpenCV
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.mask.astype(np.uint8), connectivity=8)
        no_labels = labels == 0
        label_ids = np.arange(1, num_labels)
    
        if len(label_ids) > 1:
            distances = np.zeros((self.mask.shape[0], self.mask.shape[1], len(label_ids)), dtype=np.float32)
            for i, label_id in enumerate(label_ids):
                label_mask = (labels == label_id).astype(np.uint8)
                distance_map = cv2.distanceTransform(1 - label_mask, distanceType=cv2.DIST_L2, maskSize=0)
                distances[:, :, i] = distance_map
    
            distances = np.sort(distances, axis=2)
            d1 = distances[:, :, 0]
            d2 = distances[:, :, 1]
            w = self.w0 * np.exp(-1/2*((d1 + d2) / self.sigma)**2) * no_labels
    
            if self.wc:
                class_weights = np.zeros_like(self.mask)
                for k, v in self.wc.items():
                    class_weights[self.mask == k] = v
                w = w + class_weights
        else:
            w = np.zeros_like(self.mask)
        
        return w
