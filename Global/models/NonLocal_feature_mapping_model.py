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


class Mapping_Model_with_mask(nn.Module):
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model_with_mask, self).__init__()

        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []

        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]

        self.before_NL = nn.Sequential(*model)

        if opt.NL_res:
            self.NL = networks.NonLocalBlock2D_with_mask_Res(
                mc,
                mc,
                opt.NL_fusion_method,
                opt.correlation_renormalize,
                opt.softmax_temperature,
                opt.use_self,
                opt.cosin_similarity,
            )
            print("You are using NL + Res")

        model = []
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
        self.after_NL = nn.Sequential(*model)
        
    
    def forward(self, input, mask):
        x1 = self.before_NL(input)
        del input
        x2 = self.NL(x1, mask)
        del x1, mask
        x3 = self.after_NL(x2)
        del x2

        return x3

class Mapping_Model_with_mask_2(nn.Module): ## Multi-Scale Patch Attention
    def __init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None):
        super(Mapping_Model_with_mask_2, self).__init__()

        norm_layer = networks.get_norm_layer(norm_type=norm)
        activation = nn.ReLU(True)
        model = []

        tmp_nc = 64
        n_up = 4

        for i in range(n_up):
            ic = min(tmp_nc * (2 ** i), mc)
            oc = min(tmp_nc * (2 ** (i + 1)), mc)
            model += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]

        for i in range(2):
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

        print("Mapping: You are using multi-scale patch attention, conv combine + mask input")

        self.before_NL = nn.Sequential(*model)

        if opt.mapping_exp==1:
            self.NL_scale_1=networks.Patch_Attention_4(mc,mc,8)

        model = []
        for i in range(2):
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

        self.res_block_1 = nn.Sequential(*model)

        if opt.mapping_exp==1:
            self.NL_scale_2=networks.Patch_Attention_4(mc,mc,4)

        model = []
        for i in range(2):
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
        
        self.res_block_2 = nn.Sequential(*model)
        
        if opt.mapping_exp==1:
            self.NL_scale_3=networks.Patch_Attention_4(mc,mc,2)
        # self.NL_scale_3=networks.Patch_Attention_2(mc,mc,2)

        model = []
        for i in range(2):
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
        self.after_NL = nn.Sequential(*model)
        
    
    def forward(self, input, mask):
        x1 = self.before_NL(input)
        x2 = self.NL_scale_1(x1,mask)
        x3 = self.res_block_1(x2)
        x4 = self.NL_scale_2(x3,mask)
        x5 = self.res_block_2(x4)
        x6 = self.NL_scale_3(x5,mask)
        x7 = self.after_NL(x6)
        return x7

    def inference_forward(self, input, mask):
        x1 = self.before_NL(input)
        del input
        x2 = self.NL_scale_1.inference_forward(x1,mask)
        del x1
        x3 = self.res_block_1(x2)
        del x2
        x4 = self.NL_scale_2.inference_forward(x3,mask)
        del x3
        x5 = self.res_block_2(x4)
        del x4
        x6 = self.NL_scale_3.inference_forward(x5,mask)
        del x5
        x7 = self.after_NL(x6)
        del x6
        return x7   