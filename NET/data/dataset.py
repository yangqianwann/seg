#!/usr/bin/env deeplearning
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:54:42 2020

@author: yangqianwan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import torch.nn
from torchvision import transforms
import cv2

class MyDataset(data.Dataset):
    def __init__(self, transform=None):
        self.input_images = np.load('/Users/yangqianwan/Desktop/lab/brain_tumor_dataset/brain-tumor-mri-dataset/brain_tumor_dataset/images.npy')        
        self.mask =np.load('/Users/yangqianwan/Desktop/lab/brain_tumor_dataset/brain-tumor-mri-dataset/brain_tumor_dataset/masks.npy')
        self.mask=self.mask.astype(int)
        self.labels =np.load('/Users/yangqianwan/Desktop/lab/brain_tumor_dataset/brain-tumor-mri-dataset/brain_tumor_dataset/labels.npy')
        self.transform = transform
        self.n_class   = 2
    def __getitem__(self,index):
        #image = self.input_images[idx]
        #mask = self.target_masks[idx]
        image=torch.from_numpy(np.expand_dims(np.array(self.input_images),axis=0)).float()
#        image=torch.unsqueeze(image, 0)
        mask=torch.from_numpy(np.array(self.mask)).long()
        image, mask = image[:,index,:,:], mask[index]
#        if self.transform:
#            image = self.transform(image)
#            mask = self.transform(mask)
        # create one-hot encoding
        h, w = mask.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][mask == c] = 1
        sample = {'X': image, 'Y': target, 'l': mask}
        return sample
    def __len__(self):
        return len(self.input_images)


#trans = transforms.Compose([
#    transforms.ToTensor(),
#])

#whole_set=MyDataset(transform = trans)
#length=len(whole_set)
#train_size,validate_size=2450,len(whole_set)-2450
#train_set,validate_set=data.random_split(whole_set,[train_size,validate_size])
#image_datasets = {
#    'train': train_set, 'val': validate_set
#}
#
#batch_size = 4
#dataloaders = {
#    'train': data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
#    'val': data.DataLoader(validate_set, 2, shuffle=True, num_workers=0)
#}
##print(image_datasets.dataloaders)
#dataset_sizes = {
#    x: len(image_datasets[x]) for x in image_datasets.keys()
#}

#print(dataset_sizes)


