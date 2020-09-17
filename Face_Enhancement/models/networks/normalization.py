# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


def get_nonspade_norm_layer(opt, norm_type="instance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith("spectral"):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "sync_batch":
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError("normalization layer %s is not recognized" % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, opt):
        super().__init__()

        assert config_text.startswith("spade")
        parsed = re.search("spade(\D+)(\d)x\d", config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.opt = opt
        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "syncbatch":
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError("%s is not a recognized param-free norm type in SPADE" % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2

        if self.opt.no_parsing_map:
            self.mlp_shared = nn.Sequential(nn.Conv2d(3, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        else:
            self.mlp_shared = nn.Sequential(
                nn.Conv2d(label_nc + 3, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
            )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, degraded_image):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        degraded_face = F.interpolate(degraded_image, size=x.size()[2:], mode="bilinear")

        if self.opt.no_parsing_map:
            actv = self.mlp_shared(degraded_face)
        else:
            actv = self.mlp_shared(torch.cat((segmap, degraded_face), dim=1))
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out
