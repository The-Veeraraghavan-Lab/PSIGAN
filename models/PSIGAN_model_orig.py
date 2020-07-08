# -*- coding: utf-8 -*-
"""
Created: Nov 2018

@author: Jue Jiang

Date modified: July 7 2020
@author: Harini Veeraraghavan

"""

# -*- coding: utf-8 -*-

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

import copy


class PSIGAN(BaseModel):
    def name(self):
        return 'PSIGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size) # input A
        self.input_B = self.Tensor(nb, opt.output_nc, size, size) # input B
        self.input_A_y = self.Tensor(nb, opt.output_nc, size, size) # input B
        self.input_B_y = self.Tensor(nb, opt.output_nc, size, size) # input B
        self.input_A=self.input_A.cuda()
        self.input_B=self.input_B.cuda()
        self.input_A_y=self.input_A_y.cuda()
        self.input_B_y=self.input_B_y.cuda()

        self.test_A = self.Tensor(nb, opt.output_nc, size, size) # input B
        self.test_AB = self.Tensor(nb, opt.output_nc, size, size) # input B        
        self.test_A_y = self.Tensor(nb, opt.output_nc, size, size) # input B            
       
        self.num_organ=1+4 # background + organ number
        self.netG_A = networks.define_G(1, 1,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(1, 1,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        if not self.isTrain:
            use_sigmoid = True
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_A_local = networks.define_D(2, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)      
                                                                                                
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            
            # Split-Segmentor
            # EM
            self.netSeg_B_encode=networks.get_Unet_encode_fake(1,self.num_organ,opt.init_type,self.gpu_ids)
            #ECM
            self.netSeg_B_att_encode=networks.get_Unet_encode_real(1,self.num_organ,opt.init_type,self.gpu_ids) # here is the only attention
            #DE
            self.netSeg_B_share_decode=networks.get_Unet_share_decode(1,self.num_organ,opt.init_type,self.gpu_ids) # here is the only attention


        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch

            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netSeg_A,'Seg_A',which_epoch)
                self.load_network(self.netSeg_B,'Seg_B',which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_B_local_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers

            # opt_vae = optim.Adam(list(vae.parameters())+list(d_seg.parameters()), lr=vae_learning_rate, betas=vae_betas)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            self.optimizer_D_A_local = torch.optim.Adam(self.netD_A_local.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)  
                     
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            
            self.optimizer_Seg_B_encode = torch.optim.Adam(self.netSeg_B_encode.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            self.optimizer_Seg_B_att_encode = torch.optim.Adam(self.netSeg_B_att_encode.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)
            self.optimizer_Seg_B_shared_decode = torch.optim.Adam(self.netSeg_B_share_decode.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),amsgrad=True)

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_A_local)         
                               
            self.optimizers.append(self.optimizer_D_B)
            
            self.optimizers.append(self.optimizer_Seg_B_encode)        
            self.optimizers.append(self.optimizer_Seg_B_att_encode)          
            self.optimizers.append(self.optimizer_Seg_B_shared_decode)                    
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            
            networks.print_network(self.netSeg_B_encode)      
            networks.print_network(self.netSeg_B_att_encode)      
            networks.print_network(self.netSeg_B_share_decode)                      
        print('-----------------------------------------------')
    def set_test_input(self,input):
        input_A1=input[0]
        self.test_A,self.test_A_y=torch.split(input_A1, input_A1.size(0), dim=1)    

        
    def net_G_A_load_weight(self,weight):
        self.load_network(self.netG_A, 'G_A', weight)

    def net_D_A_load_weight(self,weight):
        self.load_network(self.netD_A, 'D_A', weight)

    def net_D_B_load_weight(self,weight):
        self.load_network(self.netD_B, 'D_B', weight)

    def net_share_load_weight(self,weight):
        self.load_network(self.netSeg_B_encode, 'Seg_B_encode', weight)
        self.load_network(self.netSeg_B_att_encode, 'Seg_B_att_encode', weight)
        self.load_network(self.netSeg_B_share_decode  , 'Seg_B_shared_decode', weight)

    def get_curr_lr(self):
        self.cur_lr=self.optimizer_Seg_A.param_groups[0]['lr'] 

    def net_G_A2B_test(self):
        self.fake_A2B=self.netG_A(self.test_A)
        self.fake_A2B=self.fake_A2B.data
        self.fake_A2B_A_img, output_fakeA2B=self.tensor2im_jj(self.fake_A2B)
        return self.fake_A2B_A_img,output_fakeA2B

    def load_pre_trained_net(self,weight):
        self.load_network(self.netD_A, 'D_A', weight)
        self.load_network(self.netD_B, 'D_B', weight)
        self.load_network(self.netSeg_A,'Seg_A',weight)
        self.load_network(self.netSeg_B,'Seg_B',weight)  
        self.load_network(self.netG_A,'G_A',weight)     
        self.load_network(self.netG_B,'G_B',weight)   
        self.load_network(self.netB_attention,'Seg_B',weight)   

    def set_test_input(self,input):
        input_A1=input[0]
        self.test_A,self.test_A_y=torch.split(input_A1, input_A1.size(0), dim=1)    
        self.test_A=self.test_A.float()

    def set_test_input_map(self,input):
        #input_A1=input[0]
        self.test_A=input[:,0,:,:].view(1,1,256,256)
        self.test_A_y=input[:,1,:,:].view(1,1,256,256)#,self.test_A_y=torch.split(input_A1, input_A1.size(0), dim=1)    
        self.test_A=self.test_A.float()


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'

        input_A1=input[0]
       
        input_A11,input_A12=torch.split(input_A1, input_A1.size(1)//2, dim=1)  #mt_BN use
        
        input_B1=input[1]
        
        input_B11,input_B12=torch.split(input_B1, input_B1.size(1)//2, dim=1)  #mt_BN use        
         
        self.input_A.resize_(input_A11.size()).copy_(input_A11)
        self.input_B.resize_(input_B11.size()).copy_(input_B11)
        

        self.input_A_y.resize_(input_A12.size()).copy_(input_A12)
        self.input_B_y.resize_(input_B12.size()).copy_(input_B12)
        
    
    def cal_segmentation_loss(self,input, target):
        ## compute negative log likelihood loss
        n, c, h, w = input.size()
        input=input.float()
        log_p = F.log_softmax(input,dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(target.numel())
        target=target.long()
        loss = F.nll_loss(log_p, target, weight=None, size_average=True) 
        size_average=True
        if size_average:
            loss /= float(target.numel())

        return loss

        
    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_A_y=Variable(self.input_A_y)
        self.real_B = Variable(self.input_B)
        self.real_B_y = Variable(self.input_B_y)      
        #self.real_C = Variable(self.input_C)
        #self.real_C_y = Variable(self.input_C_y)            
    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data




    def get_image_paths(self):
        return self.image_paths

    
    def cal_seg_loss (self,netSeg,pred,gt):

        self.pred=netSeg(pred)

        lmd=self.opt.SegLambda_B    
        seg_loss=lmd*self.cal_segmentation_loss(self.pred,gt)

        return seg_loss

    def cal_seg_loss_en_decode (self,netSeg_encode,netSeg_decode,pred,gt):

        x1,x2,x3,x4,x5=netSeg_encode(pred)
        self.pred=netSeg_decode(x1,x2,x3,x4,x5)

        lmd=self.opt.SegLambda_B    
        seg_loss=lmd*self.cal_segmentation_loss(self.pred,gt)
        #seg_loss=lmd*torch.nn.functional.binary_cross_entropy(self.pred,gt)
        return seg_loss


    def backward_Real_MRI_Seg(self,netSeg,img,gt):
        lmd=self.opt.SegLambda_B    
        seg_loss=self.cal_seg_loss(netSeg,img,gt)

        return seg_loss

    def backward_Seg(self,netSeg,img,gt,img_A,img_AB):
        

        feature_loss=0
                
        lmd=self.opt.SegLambda_B    
        seg_loss=self.cal_seg_loss(netSeg,img,gt)
        
        
        return seg_loss,feature_loss


    def backward_Seg_ct_conca_fmri(self,netSeg,img,gt,img_A,img_AB):
        
        # cal feature loss
        if self.opt.use_feature_loss == 1:
            feature_loss=self.cal_feature_loss(img_A,img_AB) # no need add seg stream in the arg
        else:
            feature_loss=0
                
        # cal seg loss
        #self.pred=netSeg(img)
        lmd=self.opt.SegLambda_B    
        seg_loss=self.cal_seg_loss(netSeg,img,gt)
        if self.opt.use_feature_loss  == 1 :
            total_loss=seg_loss+feature_loss
        else:
            total_loss=seg_loss
        #total_loss.backward()

        return seg_loss,feature_loss

    def backward_Seg_B_stream(self):
        gt_A=self.real_A_y # gt 
        img_AB=self.netG_A(self.real_A) # img_AB/fake_B
        img_mri=self.real_C
        img_mri_y=self.real_C_y
        seg_loss_AB,feature_loss_AB=self.backward_Seg(self.netSeg_B,img_AB,gt_A,self.real_A,img_AB)
        seg_loss_real_B=self.backward_Real_MRI_Seg(self.netSeg_B,img_mri,img_mri_y)        
        if self.opt.use_feature_loss == 1:
            print ('segmentation loss in B:', seg_loss_AB.data[0]/self.opt.SegLambda_B, 'feature loss in B', feature_loss_AB.data[0]/self.opt.SegLambda_B)
        else:
            print ('segmentation loss in B:', seg_loss_AB.data[0])
        #self.seg_loss_AB=seg_loss_AB.data[0]


    def backward_Seg_A_stream(self):
        gt_A=self.real_A_y # gt 
        img_A=self.real_A # gt
        img_AB=self.netG_A(self.real_A) # gt
        seg_loss_A,feature_loss_A=self.backward_Seg(self.netSeg_A,img_A,gt_A,self.real_A,img_AB)

        if self.opt.use_feature_loss == 1:
            print ('segmentation loss in A:', seg_loss_A.data[0], 'feature loss in A', feature_loss_A.data[0])
        else:
            print ('segmentation loss in A:', seg_loss_A.data[0])        
        
        #print (seg_loss_A.data[0])
        #self.seg_loss_A=seg_loss_A.data[0]

    

    def backward_segmentation_B_stream(self):
        gt_A=self.real_A_y # gt 
        img_A=self.real_A # gt
        img_AB=self.netG_A(self.real_A) # gt
        img_A_AB=torch.cat((img_A,img_AB),1)

        img_mri=self.real_B
        img_mri_y=self.real_B_y
        
        seg_loss_AB=self.cal_seg_loss_en_decode(self.netSeg_B_encode,self.netSeg_B_share_decode,img_AB,gt_A)  # MRI_seg constraint
        
        seg_loss_Attention_B=self.cal_seg_loss_en_decode(self.netSeg_B_att_encode,self.netSeg_B_share_decode,self.fake_B,gt_A)  
        
        self.seg_loss_Attention_fake_B=seg_loss_Attention_B.item()#.data[0]

        total_loss=seg_loss_AB+seg_loss_Attention_B
        total_loss.backward()
        self.seg_loss_AB_=seg_loss_AB.item()#.data[0]

    
    def backward_Seg_attention_1(self,netSeg,img,gt):
        
        #self.pred=netSeg(img)
        
        seg_loss=self.cal_seg_loss_attention(netSeg,img,gt)
        return seg_loss


    def cal_seg_loss_attention (self,netSeg,pred,gt):
        #print (pred.size())
        self.pred=netSeg(pred)
        if len(self.pred)>1:
            self.pred=self.pred[2]        
        #print (self.pred.size())
        lmd=self.opt.SegLambda_B    
        #gt[gt>0]=1
        #seg_loss=self.dice_loss_binary(self.pred,gt)
        seg_loss=self.dice_loss(self.pred,gt)
        #seg_loss=lmd*torch.nn.functional.binary_cross_entropy(self.pred,gt)
        return seg_loss

    def dice_loss_binary(self,input, target):
        smooth = 1.
        #print (input.size())
        #print (target.size())
        iflat = input.view(-1)
        tflat = target.view(-1)
        #print (iflat)
        #print (tflat)
        iflat=iflat.cuda()
        tflat=tflat.cuda()
        intersection = (iflat * tflat).sum()
        
        #print ('intersection is:',intersection)
        #print ('iflat is:',iflat.sum())
        #print ('tflat is:',tflat.sum())
        return ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D
    
    def backward_D_basic_A(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        #loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B) # same to self.fake_B_local_pool
        
        loss_D_A = self.backward_D_basic_A(self.netD_A, self.real_B, fake_B)
        Total_loss_DA=loss_D_A
        self.loss_D_A = loss_D_A.item()#.data[0]
        self.fake_B_in_D=fake_B.data[0]
        return Total_loss_DA


    def backward_D_basic_A_local(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        #loss_D.backward()
        return loss_D        

    def backward_D_A_local(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        

        '''
        Here use SM for the structure discriminator update
        '''
        x1,x2,x3,x4,x5=self.netSeg_B_att_encode(fake_B) # generate fake_B segmentation
        att_map_fake_B=self.netSeg_B_share_decode(x1,x2,x3,x4,x5)

        #att_map_fake_B=self.netB_attention(fake_B)
        #_,_,att_map_fake_B=self.netSeg_B(fake_B)
        if len(att_map_fake_B)>1:
            att_map_fake_B=att_map_fake_B[2]

        att_map_fake_B=F.softmax(att_map_fake_B, dim=1)

        att_map_fake_B_1=att_map_fake_B[:,0,:,:] 
        att_map_fake_B_struct=1-att_map_fake_B_1
        

        #att_map_fake_B_parotid=att_map_fake_B_1+att_map_fake_B_2
        att_map_fake_B_struct=att_map_fake_B_struct.view(1,1,256,256)
        fake_B_struct_cat_att=torch.cat((fake_B,att_map_fake_B_struct),1)
       
        
        x1,x2,x3,x4,x5=self.netSeg_B_att_encode(self.real_B) # generate fake_B segmentation
        att_map_real_B=self.netSeg_B_share_decode(x1,x2,x3,x4,x5)

        att_map_real_B=F.softmax(att_map_real_B, dim=1)
        att_map_real_B_1=att_map_real_B[:,0,:,:] 
        att_map_real_B_struct=1-att_map_real_B_1

        
        att_map_real_B_struct=att_map_real_B_struct.view(1,1,256,256)

        real_B_struct_cat_att=torch.cat((self.real_B,att_map_real_B_struct),1)

        loss_D_A_local_struct = self.backward_D_basic_A_local(self.netD_A_local, real_B_struct_cat_att, fake_B_struct_cat_att)

        Total_loss_DA=loss_D_A_local_struct

        self.loss_D_A_local_struct=loss_D_A_local_struct.item()
 
        self.att_map_real_B_struct=att_map_real_B_struct.data
        self.att_map_fake_B_struct=att_map_fake_B_struct.data
 
        return Total_loss_DA

    def backward_D_A_global_and_local(self):
        #Global intensity discriminator
        DA_global=self.backward_D_A()
        #Structure discriminator
        DA_local=self.backward_D_A_local()
        D_A_loss_total=DA_global+self.opt.local_D_weight*DA_local
        D_A_loss_total.backward()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()#.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()#.data[0]
            self.loss_idt_B = loss_idt_B.item()#.data[0]
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0


        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        
        # local D
        fake_B_local=fake_B#+1 # change from -1---1 to 0-2
        ## OBSOLETE CODE
        #fake_B_local_submand=fake_B#+1 # change from -1---1 to 0-2
        #fake_B_local_mandible=fake_B#+1 # change from -1---1 to 0-2
        #real_B_local=self.real_B+1  #change from -1---1 to 0-2
        
        #Here use the joint distribution of segmentation and image for the generator update. Key
        x1,x2,x3,x4,x5=self.netSeg_B_encode(fake_B) # generate fake_B segmentation
        att_fake_B=self.netSeg_B_share_decode(x1,x2,x3,x4,x5)
        att_fake_B=F.softmax(att_fake_B, dim=1) # the softmax activation is used to get the probabilistic segmentation
        
        att_fake_B_1=att_fake_B[:,0,:,:] 
        att_map_fake_B=1-att_fake_B_1
        
        att_map_fake_B=att_map_fake_B.view(1,1,256,256)
        fake_B_local=fake_B_local.view(1,1,256,256)
        fake_B_local_cat_att=torch.cat((fake_B_local,att_map_fake_B),1)

        pred_fake_local = self.netD_A_local(fake_B_local_cat_att)

        loss_G_A_local = self.criterionGAN(pred_fake_local, True) # the adv local on G
        

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        seg_loss_B=self.cal_seg_loss_en_decode(self.netSeg_B_encode,self.netSeg_B_share_decode,fake_B,self.real_A_y) # the seg_B_loss 


        # combined loss
        #Here is the adversarial loss for generator including the structure generator
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B+seg_loss_B+self.opt.local_D_weight*loss_G_A_local 
        loss_G.backward()

        fake_B_and_local=torch.cat((fake_B,fake_B_local),2)

        self.fake_B = fake_B.data
        self.fake_B_and_local=fake_B_and_local.data
        #self.fake_B_variable=Variable(self.fake_B)
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.item()#.data[0]
        self.loss_G_B = loss_G_B.item()#.data[0]
        self.loss_cycle_A = loss_cycle_A.item()#.data[0]
        self.loss_cycle_B = loss_cycle_B.item()#.data[0]
        #self.loss_seg_A=seg_loss_A.item()#.data[0]
        self.loss_seg_B=seg_loss_B.item()#.data[0]
        self.loss_feature=0#feature_loss.data[0]
        self.loss_G_A_loca=loss_G_A_local.item()#.data[0]
        self.seg_loss_Real_B=0#seg_acc_Real_B.data[0]
        self.seg_loss_fake_B=0#seg_acc_fake_B.data[0]
        #self.att_map_fake_B_parotid=att_map_fake_B.view(1,1,256,256).data
        self.fake_B_local=fake_B_local.data
        self.loss_G_A_local=loss_G_A_local.item()#.data[0]


    def load_CT_seg_A(self, weight):
        self.load_network(self.netSeg_A,'Seg_A',weight)

    def load_Generator_A(self, weight):
        self.load_network(self.netG_A,'G_A',weight)

    def load_CT_seg_B_attention(self, weight):
        self.load_network(self.netB_attention,'Seg_B_attention',weight)

    def load_CT_seg_B_attention_seg_A(self, weight):
        self.load_network(self.netB_attention,'Seg_B',weight)


    def optimize_parameters(self):
        # forward
        self.forward()

        # G_A and G_B
        self.optimizer_G.zero_grad()
        # SegB + D_struct->G
        # G_A_struct
        self.backward_G()
        self.optimizer_G.step()
       # D_A
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_A_local.zero_grad()
        # SegB_att + G->D_struct
        
        #Key part: Global intensity and Structure discriminator
        self.backward_D_A_global_and_local()
          
        self.optimizer_D_A.step()
        self.optimizer_D_A_local.step()

        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        
        self.optimizer_Seg_B_encode.zero_grad()
        self.optimizer_Seg_B_att_encode.zero_grad()
        self.optimizer_Seg_B_shared_decode.zero_grad()

        self.backward_segmentation_B_stream()
        #G->SegB_att, G->SegB     f(segB)~~f(segB_att)
        
        self.optimizer_Seg_B_encode.step()
        self.optimizer_Seg_B_att_encode.step()
        self.optimizer_Seg_B_shared_decode.step()
  

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                 ('Seg_B',  self.loss_seg_B),('loss_G_A_loca',self.loss_G_A_loca),('seg_loss_Real_B',  self.seg_loss_Real_B),
                                 ('seg_loss_fake_B',self.seg_loss_AB_),('seg_loss_Attention_fake_B',self.seg_loss_Attention_fake_B),
                                 ('G_A_struct',self.loss_G_A_local),
                                 ('D_A_struct',  self.loss_D_A_local_struct)])

        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        real_Ay=util.tensor2im_hd_neck(self.input_A_y)

        fake_B_local=util.tensor2im(self.fake_B_local)
        real_B_local=util.tensor2im(self.real_B)
        
        #Attention map for Real MRI
        att_real_B=util.tensor2im(self.att_map_real_B_struct)
        #Segmentation probability map for pseudoMRI
        att_fake_B=util.tensor2im(self.att_map_fake_B_struct)

        

        x1,x2,x3,x4,x5=self.netSeg_B_att_encode(self.real_B) # generate fake_B segmentation
        pred_B=self.netSeg_B_share_decode(x1,x2,x3,x4,x5)

        #pred_B=self.netB_attention(self.real_B)


         
        pred_B=torch.argmax(pred_B, dim=1)
        
        x1,x2,x3,x4,x5=self.netSeg_B_att_encode(self.fake_B) # generate fake_B segmentation
        pred_B_att_fB=self.netSeg_B_share_decode(x1,x2,x3,x4,x5)



        pred_B_att_fB=torch.argmax(pred_B_att_fB, dim=1)

        pred_B_att_fB=pred_B_att_fB.view(1,1,256,256)

        pred_B=pred_B.view(1,1,256,256)        

        x1,x2,x3,x4,x5=self.netSeg_B_encode(self.fake_B) # generate fake_B segmentation
        fake_B_seg=self.netSeg_B_share_decode(x1,x2,x3,x4,x5)

        #t1,t2,fake_B_seg=self.netSeg_B(self.fake_B)
        fake_B_seg=torch.argmax(fake_B_seg, dim=1)
        fake_B_seg=fake_B_seg.view(-1,1,256,256)        
        
        pred_B=pred_B.data
        fake_B_seg=fake_B_seg.data
        pred_B_att_fB=pred_B_att_fB.data

        seg_B=util.tensor2im_hd_neck(pred_B)#() #
        seg_B_atten_fB=util.tensor2im(pred_B_att_fB) #
        fake_B_seg=util.tensor2im_hd_neck(fake_B_seg) #
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                   ('fake_B_seg', fake_B_seg),('real_B_seg', seg_B),
                                   ('real_A_GT_seg',real_Ay),('att_map_fake_B',att_fake_B),('att_map_real_B',att_real_B),
                                   ('real_B_struct',real_B_local),('fake_B_struct',fake_B_local),('fake_B_seg_attention',seg_B_atten_fB)])
        
        return ret_visuals


    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netSeg_B_att_encode, 'Seg_B_att_encode', label, self.gpu_ids)
        self.save_network(self.netSeg_B_encode, 'Seg_B_encode', label, self.gpu_ids)        
        self.save_network(self.netSeg_B_share_decode, 'Seg_B_shared_decode', label, self.gpu_ids) 
























        
