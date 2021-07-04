# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import spectral_norm

# from util.util import SwitchNorm2d
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == "spectral":
        norm_layer = spectral_norm()
    elif norm_type == "SwitchNorm":
        norm_layer = SwitchNorm2d
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: %d" % num_params)


def define_G(input_nc, output_nc, ngf, netG, k_size=3, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[], opt=None):
    
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        # if opt.self_gen:
        if opt.use_v2:
            netG = GlobalGenerator_DCDCv2(input_nc, output_nc, ngf, k_size, n_downsample_global, norm_layer, opt=opt)
        else:
            netG = GlobalGenerator_v2(input_nc, output_nc, ngf, k_size, n_downsample_global, n_blocks_global, norm_layer, opt=opt)
    else:
        raise('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, n_layers_D, opt, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, opt, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD



class GlobalGenerator_DCDCv2(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        k_size=3,
        n_downsampling=8,
        norm_layer=nn.BatchNorm2d,
        padding_type="reflect",
        opt=None,
    ):
        super(GlobalGenerator_DCDCv2, self).__init__()
        activation = nn.ReLU(True)

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, min(ngf, opt.mc), kernel_size=7, padding=0),
            norm_layer(ngf),
            activation,
        ]
        ### downsample
        for i in range(opt.start_r):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, opt.mc),
                    min(ngf * mult * 2, opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, opt.mc)),
                activation,
            ]
        for i in range(opt.start_r, n_downsampling - 1):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, opt.mc),
                    min(ngf * mult * 2, opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, opt.mc)),
                activation,
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
        mult = 2 ** (n_downsampling - 1)

        if opt.spatio_size == 32:
            model += [
                nn.Conv2d(
                    min(ngf * mult, opt.mc),
                    min(ngf * mult * 2, opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, opt.mc)),
                activation,
            ]
        if opt.spatio_size == 64:
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
        model += [
            ResnetBlock(
                min(ngf * mult * 2, opt.mc),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer,
                opt=opt,
            )
        ]
        # model += [nn.Conv2d(min(ngf * mult * 2, opt.mc), min(ngf, opt.mc), 1, 1)]
        if opt.feat_dim > 0:
            model += [nn.Conv2d(min(ngf * mult * 2, opt.mc), opt.feat_dim, 1, 1)]
        self.encoder = nn.Sequential(*model)

        # decode
        model = []
        if opt.feat_dim > 0:
            model += [nn.Conv2d(opt.feat_dim, min(ngf * mult * 2, opt.mc), 1, 1)]
        # model += [nn.Conv2d(min(ngf, opt.mc), min(ngf * mult * 2, opt.mc), 1, 1)]
        o_pad = 0 if k_size == 4 else 1
        mult = 2 ** n_downsampling
        model += [
            ResnetBlock(
                min(ngf * mult, opt.mc),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer,
                opt=opt,
            )
        ]

        if opt.spatio_size == 32:
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, opt.mc),
                    min(int(ngf * mult / 2), opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), opt.mc)),
                activation,
            ]
        if opt.spatio_size == 64:
            model += [
                ResnetBlock(
                    min(ngf * mult, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]

        for i in range(1, n_downsampling - opt.start_r):
            mult = 2 ** (n_downsampling - i)
            model += [
                ResnetBlock(
                    min(ngf * mult, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult, opt.mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=opt,
                )
            ]
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, opt.mc),
                    min(int(ngf * mult / 2), opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), opt.mc)),
                activation,
            ]
        for i in range(n_downsampling - opt.start_r, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, opt.mc),
                    min(int(ngf * mult / 2), opt.mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), opt.mc)),
                activation,
            ]
        if opt.use_segmentation_model:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(min(ngf, opt.mc), output_nc, kernel_size=7, padding=0)]
        else:
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(min(ngf, opt.mc), output_nc, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        self.decoder = nn.Sequential(*model)

    def forward(self, input, flow="enc_dec"):
        if flow == "enc":
            return self.encoder(input)
        elif flow == "dec":
            return self.decoder(input)
        elif flow == "enc_dec":
            x = self.encoder(input)
            x = self.decoder(x)
            return x


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(
        self, dim, padding_type, norm_layer, opt, activation=nn.ReLU(True), use_dropout=False, dilation=1
    ):
        super(ResnetBlock, self).__init__()
        self.opt = opt
        self.dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == "zero":
            p = self.dilation
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=1), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b : b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[
                        indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]
                    ] = mean_feat
        return outputs_mean


def SN(module, mode=True):
    if mode:
        return torch.nn.utils.spectral_norm(module)

    return module


class NonLocalBlock2D_with_mask_Res(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels,
        mode="add",
        re_norm=False,
        temperature=1.0,
        use_self=False,
        cosin=False,
    ):
        super(NonLocalBlock2D_with_mask_Res, self).__init__()

        self.cosin = cosin
        self.renorm = re_norm
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.W = nn.Conv2d(
            in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0
        )
        # for pytorch 0.3.1
        # nn.init.constant(self.W.weight, 0)
        # nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.mode = mode
        self.temperature = temperature
        self.use_self = use_self

        norm_layer = get_norm_layer(norm_type="instance")
        activation = nn.ReLU(True)

        model = []
        for i in range(3):
            model += [
                ResnetBlock(
                    inter_channels,
                    padding_type="reflect",
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=None,
                )
            ]
        self.res_block = nn.Sequential(*model)

    def forward(self, x, mask):  ## The shape of mask is Batch*1*H*W
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        if self.cosin:
            theta_x = F.normalize(theta_x, dim=2)
            phi_x = F.normalize(phi_x, dim=1)

        f = torch.matmul(theta_x, phi_x)

        f /= self.temperature

        f_div_C = F.softmax(f, dim=2)

        tmp = 1 - mask
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode="bilinear")
        mask[mask > 0] = 1.0
        mask = 1 - mask

        tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        mask *= tmp

        mask_expand = mask.view(batch_size, 1, -1)
        mask_expand = mask_expand.repeat(1, x.size(2) * x.size(3), 1)

        # mask = 1 - mask
        # mask=F.interpolate(mask,(x.size(2),x.size(3)))
        # mask_expand=mask.view(batch_size,1,-1)
        # mask_expand=mask_expand.repeat(1,x.size(2)*x.size(3),1)

        if self.use_self:
            mask_expand[:, range(x.size(2) * x.size(3)), range(x.size(2) * x.size(3))] = 1.0

        #    print(mask_expand.shape)
        #    print(f_div_C.shape)

        f_div_C = mask_expand * f_div_C
        if self.renorm:
            f_div_C = F.normalize(f_div_C, p=1, dim=2)

        ###########################

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        W_y = self.res_block(W_y)

        if self.mode == "combine":
            full_mask = mask.repeat(1, self.inter_channels, 1, 1)
            z = full_mask * x + (1 - full_mask) * W_y
        return z


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, opt, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, opt, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, opt, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[SN(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),opt.use_SN), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                SN(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),opt.use_SN),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            SN(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),opt.use_SN),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[SN(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw),opt.use_SN)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)



class Patch_Attention_4(nn.Module):  ## While combine the feature map, use conv and mask
    def __init__(self, in_channels, inter_channels, patch_size):
        super(Patch_Attention_4, self).__init__()

        self.patch_size=patch_size


        # self.g = nn.Conv2d(
        #     in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        # )

        # self.W = nn.Conv2d(
        #     in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0
        # )
        # # for pytorch 0.3.1
        # # nn.init.constant(self.W.weight, 0)
        # # nn.init.constant(self.W.bias, 0)
        # # for pytorch 0.4.0
        # nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)
        # self.theta = nn.Conv2d(
        #     in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        # )

        # self.phi = nn.Conv2d(
        #     in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        # )

        self.F_Combine=nn.Conv2d(in_channels=1025,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True)
        norm_layer = get_norm_layer(norm_type="instance")
        activation = nn.ReLU(True)

        model = []
        for i in range(1):
            model += [
                ResnetBlock(
                    inter_channels,
                    padding_type="reflect",
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=None,
                )
            ]
        self.res_block = nn.Sequential(*model)

    def Hard_Compose(self, input, dim, index):
        # batch index select
        # input: [B,C,HW]
        # dim: scalar > 0
        # index: [B, HW]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, z, mask):  ## The shape of mask is Batch*1*H*W

        x=self.res_block(z)

        b,c,h,w=x.shape

        ## mask resize + dilation
        # tmp = 1 - mask
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode="bilinear")
        mask[mask > 0] = 1.0

        # mask = 1 - mask
        # tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        # mask *= tmp
        # mask=1-mask
        ## 1: mask position 0: non-mask

        mask_unfold=F.unfold(mask, kernel_size=(self.patch_size,self.patch_size), padding=0, stride=self.patch_size)
        non_mask_region=(torch.mean(mask_unfold,dim=1,keepdim=True)>0.6).float()
        all_patch_num=h*w/self.patch_size/self.patch_size
        non_mask_region=non_mask_region.repeat(1,int(all_patch_num),1)

        x_unfold=F.unfold(x, kernel_size=(self.patch_size,self.patch_size), padding=0, stride=self.patch_size)
        y_unfold=x_unfold.permute(0,2,1)
        x_unfold_normalized=F.normalize(x_unfold,dim=1)
        y_unfold_normalized=F.normalize(y_unfold,dim=2)
        correlation_matrix=torch.bmm(y_unfold_normalized,x_unfold_normalized)
        correlation_matrix=correlation_matrix.masked_fill(non_mask_region==1.,-1e9)
        correlation_matrix=F.softmax(correlation_matrix,dim=2)

        # print(correlation_matrix)

        R, max_arg=torch.max(correlation_matrix,dim=2)

        composed_unfold=self.Hard_Compose(x_unfold, 2, max_arg)
        composed_fold=F.fold(composed_unfold,output_size=(h,w),kernel_size=(self.patch_size,self.patch_size),padding=0,stride=self.patch_size)

        concat_1=torch.cat((z,composed_fold,mask),dim=1)
        concat_1=self.F_Combine(concat_1)

        return concat_1

    def inference_forward(self,z,mask): ## Reduce the extra memory cost


        x=self.res_block(z)

        b,c,h,w=x.shape

        ## mask resize + dilation
        # tmp = 1 - mask
        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode="bilinear")
        mask[mask > 0] = 1.0
        # mask = 1 - mask
        # tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        # mask *= tmp
        # mask=1-mask
        ## 1: mask position 0: non-mask

        mask_unfold=F.unfold(mask, kernel_size=(self.patch_size,self.patch_size), padding=0, stride=self.patch_size)
        non_mask_region=(torch.mean(mask_unfold,dim=1,keepdim=True)>0.6).float()[0,0,:] # 1*1*all_patch_num

        all_patch_num=h*w/self.patch_size/self.patch_size

        mask_index=torch.nonzero(non_mask_region,as_tuple=True)[0]


        if len(mask_index)==0: ## No mask patch is selected, no attention is needed

            composed_fold=x

        else:

            unmask_index=torch.nonzero(non_mask_region!=1,as_tuple=True)[0]

            x_unfold=F.unfold(x, kernel_size=(self.patch_size,self.patch_size), padding=0, stride=self.patch_size)
            
            Query_Patch=torch.index_select(x_unfold,2,mask_index)
            Key_Patch=torch.index_select(x_unfold,2,unmask_index)

            Query_Patch=Query_Patch.permute(0,2,1)        
            Query_Patch_normalized=F.normalize(Query_Patch,dim=2)
            Key_Patch_normalized=F.normalize(Key_Patch,dim=1)

            correlation_matrix=torch.bmm(Query_Patch_normalized,Key_Patch_normalized)
            correlation_matrix=F.softmax(correlation_matrix,dim=2)


            R, max_arg=torch.max(correlation_matrix,dim=2)

            composed_unfold=self.Hard_Compose(Key_Patch, 2, max_arg)
            x_unfold[:,:,mask_index]=composed_unfold
            composed_fold=F.fold(x_unfold,output_size=(h,w),kernel_size=(self.patch_size,self.patch_size),padding=0,stride=self.patch_size)

        concat_1=torch.cat((z,composed_fold,mask),dim=1)
        concat_1=self.F_Combine(concat_1)


        return concat_1

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)




####################################### VGG Loss

from torchvision import models
class VGG19_torch(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19_torch, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss_torch(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss_torch, self).__init__()
        self.vgg = VGG19_torch().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss