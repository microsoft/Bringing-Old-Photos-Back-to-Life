# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import functools
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import math
from .NonLocal_feature_mapping_model import *


class Mapping_Model(nn.Module):
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model, self).__init__()

        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []
        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        for i in range(n_blocks):
            model += [
                networks.ResnetBlock(
                    mc,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                    dilation=opt.mapping_net_dilation,
                )
            ]

        for i in range(n_up - 1):
            ic = min(64 * (2 ** (4 - i)), mc)
            oc = min(64 * (2 ** (3 - i)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        model += [nn.Conv2d(tmp_nc * 2, tmp_nc, 3, 1, 1)]
        if opt.feat_dim > 0 and opt.feat_dim < 64:
            model += [norm_layer(tmp_nc), activation, nn.Conv2d(tmp_nc, opt.feat_dim, 1, 1)]
        # model += [nn.Conv2d(64, 1, 1, 1, 0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Pix2PixHDModel_Mapping(BaseModel):
    def name(self):
        return "Pix2PixHDModel_Mapping"

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_smooth_l1, stage_1_feat_l2):
        flags = (True, True, use_gan_feat_loss, use_vgg_loss, True, True, use_smooth_l1, stage_1_feat_l2)

        def loss_filter(g_feat_l2, g_gan, g_gan_feat, g_vgg, d_real, d_fake, smooth_l1, stage_1_feat_l2):
            return [
                l
                for (l, f) in zip(
                    (g_feat_l2, g_gan, g_gan_feat, g_vgg, d_real, d_fake, smooth_l1, stage_1_feat_l2), flags
                )
                if f
            ]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != "none" or not opt.isTrain:
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        self.netG_A = networks.GlobalGenerator_DCDCv2(
            netG_input_nc,
            opt.output_nc,
            opt.ngf,
            opt.k_size,
            opt.n_downsample_global,
            networks.get_norm_layer(norm_type=opt.norm),
            opt=opt,
        )
        self.netG_B = networks.GlobalGenerator_DCDCv2(
            netG_input_nc,
            opt.output_nc,
            opt.ngf,
            opt.k_size,
            opt.n_downsample_global,
            networks.get_norm_layer(norm_type=opt.norm),
            opt=opt,
        )

        if opt.non_local == "Setting_42" or opt.NL_use_mask:
            self.mapping_net = Mapping_Model_with_mask(
                min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc),
                opt.map_mc,
                n_blocks=opt.mapping_n_block,
                opt=opt,
            )
        else:
            self.mapping_net = Mapping_Model(
                min(opt.ngf * 2 ** opt.n_downsample_global, opt.mc),
                opt.map_mc,
                n_blocks=opt.mapping_n_block,
                opt=opt,
            )

        self.mapping_net.apply(networks.weights_init)

        if opt.load_pretrain != "":
            self.load_network(self.mapping_net, "mapping_net", opt.which_epoch, opt.load_pretrain)

        if not opt.no_load_VAE:

            self.load_network(self.netG_A, "G", opt.use_vae_which_epoch, opt.load_pretrainA)
            self.load_network(self.netG_B, "G", opt.use_vae_which_epoch, opt.load_pretrainB)
            for param in self.netG_A.parameters():
                param.requires_grad = False
            for param in self.netG_B.parameters():
                param.requires_grad = False
            self.netG_A.eval()
            self.netG_B.eval()

        if opt.gpu_ids:
            self.netG_A.cuda(opt.gpu_ids[0])
            self.netG_B.cuda(opt.gpu_ids[0])
            self.mapping_net.cuda(opt.gpu_ids[0])
        
        if not self.isTrain:
            self.load_network(self.mapping_net, "mapping_net", opt.which_epoch)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.ngf * 2 if opt.feat_gan else input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1

            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                              opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.Smooth_L1, opt.use_two_stage_mapping)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)


            self.criterionFeat = torch.nn.L1Loss()
            self.criterionFeat_feat = torch.nn.L1Loss() if opt.use_l1_feat else torch.nn.MSELoss()

            if self.opt.image_L1:
                self.criterionImage=torch.nn.L1Loss()
            else:
                self.criterionImage = torch.nn.SmoothL1Loss()


            print(self.criterionFeat_feat)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_Feat_L2', 'G_GAN', 'G_GAN_Feat', 'G_VGG','D_real', 'D_fake', 'Smooth_L1', 'G_Feat_L2_Stage_1')

            # initialize optimizers
            # optimizer G

            if opt.no_TTUR:
                beta1,beta2=opt.beta1,0.999
                G_lr,D_lr=opt.lr,opt.lr
            else:
                beta1,beta2=0,0.9
                G_lr,D_lr=opt.lr/2,opt.lr*2


            if not opt.no_load_VAE:
                params = list(self.mapping_net.parameters())
                self.optimizer_mapping = torch.optim.Adam(params, lr=G_lr, betas=(beta1, beta2))

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=D_lr, betas=(beta1, beta2))

            print("---------- Optimizers initialized -------------")

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, pair=True, infer=False, last_label=None, last_image=None):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)  

        # Fake Generation
        input_concat = input_label
        
        label_feat = self.netG_A.forward(input_concat, flow='enc')
        # print('label:')
        # print(label_feat.min(), label_feat.max(), label_feat.mean())
        #label_feat = label_feat / 16.0

        if self.opt.NL_use_mask:
            label_feat_map=self.mapping_net(label_feat.detach(),inst)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())
        
        fake_image = self.netG_B.forward(label_feat_map, flow='dec')
        image_feat = self.netG_B.forward(real_image, flow='enc')

        loss_feat_l2_stage_1=0
        loss_feat_l2 = self.criterionFeat_feat(label_feat_map, image_feat.data) * self.opt.l2_feat
            

        if self.opt.feat_gan:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(label_feat.detach(), label_feat_map, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

            # Real Detection and Loss        
            pred_real = self.discriminate(label_feat.detach(), image_feat)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netD.forward(torch.cat((label_feat.detach(), label_feat_map), dim=1))        
            loss_G_GAN = self.criterionGAN(pred_fake, True)  
        else:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

            # Real Detection and Loss  
            if pair:      
                pred_real = self.discriminate(input_label, real_image)
            else:
                pred_real = self.discriminate(last_label, last_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
            loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss and pair:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    tmp = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    loss_G_GAN_Feat += D_weights * feat_weights * tmp
        else:
            loss_G_GAN_Feat = torch.zeros(1).to(label.device)
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat if pair else torch.zeros(1).to(label.device)

        smooth_l1_loss=0
        if self.opt.Smooth_L1:
            smooth_l1_loss=self.criterionImage(fake_image,real_image)*self.opt.L1_weight


        return [ self.loss_filter(loss_feat_l2, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake,smooth_l1_loss,loss_feat_l2_stage_1), None if not infer else fake_image ]


    def inference(self, label, inst):

        use_gpu = len(self.opt.gpu_ids) > 0
        if use_gpu:
            input_concat = label.data.cuda()
            inst_data = inst.cuda()
        else:
            input_concat = label.data
            inst_data = inst

        label_feat = self.netG_A.forward(input_concat, flow="enc")

        if self.opt.NL_use_mask:
            label_feat_map = self.mapping_net(label_feat.detach(), inst_data)
        else:
            label_feat_map = self.mapping_net(label_feat.detach())

        fake_image = self.netG_B.forward(label_feat_map, flow="dec")
        return fake_image


class InferenceModel(Pix2PixHDModel_Mapping):
    def forward(self, label, inst):
        return self.inference(label, inst)

