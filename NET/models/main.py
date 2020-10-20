#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:09:18 2020

@author: yangqianwan
"""

import sys
sys.path.append(r'/Users/yangqianwan/Desktop/lab/NET/data')
from unet_2d import unet_2d
#import sys
#sys.path.append(r'/Users/yangqianwan/Desktop/lab/NET')
#from config import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from dataset import MyDataset
from torch.utils import data
from torch.autograd import Variable

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))
#trans = transforms.Compose([
#    transforms.ToTensor(),
#])

whole_set=MyDataset()
length=len(whole_set)
train_size=2
train_size,validate_size=train_size,len(whole_set)-train_size
train_set,validate_set=data.random_split(whole_set,[train_size,validate_size])
image_datasets = {
    'train': train_set, 'val': validate_set
}

batch_size = 2
train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(validate_set, batch_size=1, shuffle=True)
model = unet_2d()
# pixel accuracy and mIOU list 
pixel_acc_list = []
mIOU_list = []
#batch_size = 4
#dataloaders = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
## parameters for Solver-Adam in this example
batch_size = 6 #
lr         = 1e-4    # achieved besty results 
step_size  = 100 # Won't work when epochs <=100
gamma      = 0.5 # 
epochs=2
# Observe that all parameters are being optimized
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  
#loss_meter=AverageMeter()
def train():
    for epoch in range(epochs):
        scheduler.step()
        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data.item()))        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    
    val(epoch)
        
    highest_pixel_acc = max(pixel_acc_list)
    highest_mIOU = max(mIOU_list)        
    
    highest_pixel_acc_epoch = pixel_acc_list.index(highest_pixel_acc)
    highest_mIOU_epoch = mIOU_list.index(highest_mIOU)
    
    print("The highest mIOU is {} and is achieved at epoch-{}".format(highest_mIOU, highest_mIOU_epoch))
    print("The highest pixel accuracy  is {} and is achieved at epoch-{}".format(highest_pixel_acc, highest_pixel_acc_epoch))
def save_result_comparison(input_np, output_np):
    means     = np.array([103.939, 116.779, 123.68]) / 255.
    
    global global_index
    
    original_im = np.zeros((512,512,1))    
    original_im[:,:,0] = input_np[0,0,:,:]        
    original_im[:,:,0] = original_im[:,:,0] + means[0]      
    original_im[:,:,0] = original_im[:,:,0]*255.0   
    im_seg = np.zeros((512,512,1))

    # the following version is designed for 11-class version and could still work if the number of classes is fewer.
    for i in range(512):
        for j in range(512):
            if output_np[i,j] == 0:
                im_seg[i,j,:] = 128
            elif output_np[i,j] == 1:  
                im_seg[i,j,:] = 0
   
                    
    # horizontally stack original image and its corresponding segmentation results     
    hstack_image = np.hstack((original_im, im_seg))             
    new_im = Image.fromarray(np.uint8(hstack_image))
    
    file_name = folder_to_save_validation_result + str(global_index) + '.jpg'
        
    global_index = global_index + 1
        
    new_im.save(file_name)       
    
def val(epoch):
    model.eval()
    total_ious = []
    pixel_accs = []
                    
    
    for iter, batch in enumerate(val_loader): ## batch is 1 in this case
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])        

        output = model(inputs)                                
        
        # only save the 1st image for comparison
        if iter == 0:
            print('---------iter={}'.format(iter))
            # generate images
            images = output.data.max(1)[1].cpu().numpy()[:,:,:]
            image = images[0,:,:]        
            save_result_comparison(batch['X'], image)
                            
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape                
        pred = output.transpose(0, 2, 3, 1).reshape(-1, num_class).argmax(axis=1).reshape(N, h, w)        
        target = batch['l'].cpu().numpy().reshape(N, h, w)

        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    
    global pixel_acc_list
    global mIOU_list
    
    pixel_acc_list.append(pixel_accs)
    mIOU_list.append(np.nanmean(ious))

# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(num_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

train()

     
#for epoch in range(1):
#    running_loss = 0.0
#    for inputs, labels in dataloaders:
#        inputs = inputs.float()
#        labels = labels.long()
#        #print(inputs.shape)
#        # zero the parameter gradients
#        optimizer.zero_grad()
#        outputs = model(inputs)
#        loss = criterion(outputs, labels)
#        loss.backward()
#        optimizer.step()
#        running_loss += loss.item()
#        if i % 2000 == 1999:
#            print('[%d, %5d] loss: %.3f' %
#            (epoch + 1, i + 1, running_loss / 2000))
#            running_loss = 0.0
#print('Finished Training')
