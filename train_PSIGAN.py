# -*- coding: utf-8 -*-
"""
Created on Jan 05 2019
@author: Jue Jiang
Date modified: July 07 2020
@author: Harini Veeraraghavan

Wrapper code for PSIGAN segmentor training 

"""

# -*- coding: utf-8 -*-

import time
import torch.utils.data
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import save_image
import numpy as np
import itertools
from PIL import Image
opt = TrainOptions().parse()
from skimage.measure import regionprops
import os
import scipy.io as sio
import os.path
import os

## normalize the CT images by clipping
def normalize_data(data):
    data[data<24]=24
    data[data>1524]=1524
    
    data=data-24
    
    data=data*2./1500 - 1

    return  (data)

## normalize MRI images with clipping
def normalize_data_MRI(data):

    # The MRI max value could changed according to your data max value
    data[data>1000]=1000

    data=data*2./1000 - 1

    return  (data)
    

    
model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0
print ('Loading data......')


#Data format [B,1,H,W]
SOURCE_IMG_NAME = './datasets/CT_img_sample.npy'
SOURCE_LABEL_NAME ='./datasets/CT_label_sample.npy'

TARGET_IMG_NAME = './datasets/T1w_img_sample.npy'
TARGET_LABEL_NAME = './datasets/T1w_label_sample.npy'

# Load your source image and source label
source_img=np.load(SOURCE_IMG_NAME)
source_label=np.load(SOURCE_LABEL_NAME)

# Load your target  image and targe label. Note target label is not used for supervised training.
# But it could be used for your validation accuracy
target_img=np.load(TARGET_IMG_NAME)
target_label=np.load(TARGET_LABEL_NAME)



source_img=normalize_data(source_img)
target_img=normalize_data_MRI(target_img)



source_img_cat=np.concatenate((source_img, source_label), 1)
target_img_cat=np.concatenate((target_img, target_label), 1)



train_loader_source=torch.utils.data.DataLoader(source_img_cat,#images_a_y_jj,  ## data + label 
                                         batch_size=opt.batchSize,
                                         shuffle=True,
                                         num_workers=4)       



train_loader_target=torch.utils.data.DataLoader(target_img_cat,#images_b_y_jj,  ## data + label 
                                         batch_size=opt.batchSize,
                                           shuffle=True,
                                           num_workers=4)

import numpy


device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i_iter, (data) in enumerate(zip(train_loader_source,train_loader_target)):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize


        model.set_input(data)
        model.optimize_parameters()
        
        if total_steps % opt.display_freq == 0:
        
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        

    
    model.update_learning_rate()

