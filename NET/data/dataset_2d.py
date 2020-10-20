#!/usr/bin/env python
#coding:utf8
import nibabel as nib
import os
#import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import  transforms as T
from PIL import Image
from torch.utils import data
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

###
class Brats17(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False, val = False):
        self.test = test
        self.train = train
        self.val = val

        if self.train:
            self.root = '/userhome/GUOXUTAO/temp/train/data0/'
            self.folderlist = os.listdir(os.path.join(self.root))
        elif self.val:
            self.root = '/userhome/GUOXUTAO/temp/val/data/'
            self.folderlist = os.listdir(os.path.join(self.root))
        elif self.test:
            self.root = ''
            self.folderlist = os.listdir(os.path.join(self.root))

    def __getitem__(self,index):
          
        if self.train:                            
            if 1 > 0 :
                path = self.root
                img = np.load(os.path.join(path,self.folderlist[index]))
                img = np.asarray(img)
                index_x = np.random.randint(65,175,size=1)
                index_y = np.random.randint(65,175,size=1)
                index_z = np.random.randint(65,90,size=1)
                #print(index_x,index_y,index_z)

                img_in = img[:,index_x[0]-64:index_x[0]+64,index_y[0]-64:index_y[0]+64,index_z[0]-64:index_z[0]+64]            
                img_out = img_in[0:4,:,:,:].astype(float)
                label_out = img_in[4,:,:,:].astype(float)
                #print(img_in.shape)
                img = torch.from_numpy(img_out).float()        
                label = torch.from_numpy(label_out).long()
                
        elif self.val:
            path = self.root
            img = np.load(os.path.join(path,self.folderlist[index]))
            img = np.asarray(img)
            img_out = img[0:4,:,:,:].astype(float)
            label_out = img[4,:,:,:].astype(float)
            #print(img_in.shape)
            img = torch.from_numpy(img_out).float()     
            label = torch.from_numpy(label_out).long()
        else:
            print('###$$$$$$$$$$$$$$$$$$$^^^^^^^^^^^^^')     

        return img, label

    def __len__(self):
        return len(self.folderlist)







