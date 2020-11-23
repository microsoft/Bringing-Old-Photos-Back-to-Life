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

        if opt.non_local == "Setting_42":
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
        self.load_network(self.mapping_net, "mapping_net", opt.which_epoch)

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

