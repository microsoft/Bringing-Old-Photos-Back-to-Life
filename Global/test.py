# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2
import onnxruntime

from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
import util.util as util


def encode_input(opt, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
    if opt.label_nc == 0:
        input_label = label_map.data.cuda()
    else:
        # create one-hot vector for label map
        size = label_map.size()
        oneHot_size = (size[0], opt.label_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
        if self.opt.data_type == 16:
            input_label = input_label.half()

    # get edges from instance map
    if not opt.no_instance:
        inst_map = inst_map.data.cuda()
        edge_map = self.get_edges(inst_map)
        input_label = torch.cat((input_label, edge_map), dim=1)
    input_label = Variable(input_label, volatile=infer)

    # real images for training
    if real_image is not None:
        real_image = Variable(real_image.data.cuda())

    return input_label, inst_map, real_image, feat_map


def data_transforms(img, method=Image.BILINEAR, scale=False):

    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def irregular_hole_synthesize(img, mask):

    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img


def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"


def make_onnx_files(model, device="cuda"):
    os.makedirs("onnx_models", exist_ok=True)
    if not os.path.exists("onnx_models/netG_A_encoder.onnx"):
        dummy_input = torch.randn(1, 3, 512, 512, requires_grad=True, device=device)
        torch.onnx.export(
            model.netG_A.encoder,
            dummy_input,
            "onnx_models/netG_A_encoder.onnx",
            opset_version=11,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = {'input': [0, 2, 3], 'output': [0, 2, 3]}
        )

    if not os.path.exists("onnx_models/netG_B_decoder.onnx"):
        dummy_input = torch.randn(1, 64, 128, 128, requires_grad=True, device=device)
        torch.onnx.export(
            model.netG_B.decoder,
            dummy_input,
            "onnx_models/netG_B_decoder.onnx",
            opset_version=11,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = {
                'input': [0, 2, 3], 'output': [0, 2, 3]
            }
        )

    if not os.path.exists("onnx_models/mapping_net.onnx"):
        dummy_input = (
            torch.randn(1, 64, 128, 128, requires_grad=True, device=device),
            torch.randn(1, 1, 928, 704, requires_grad=True, device=device)
        )
        torch.onnx.export(
            model.mapping_net,
            dummy_input,
            "onnx_models/mapping_net.onnx",
            opset_version=11,
            input_names = ['input', 'mask'],
            output_names = ['output'],
            dynamic_axes = {
               'input': [0, 2, 3], 'mask': [0, 2, 3], 'output': [0, 2, 3]
            }
        )

"""
RuntimeError: Given groups=1, weight of size [64, 64, 3, 3],
expected input[1, 3, 130, 130] to have 64 channels,
but got 3 channels instead
Finish Stage 1 ...
"""

def to_numpy(tensor):
    if isinstance(tensor, list):
        return np.array(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_onnx_sessions():
    netG_A_encoder = onnxruntime.InferenceSession("onnx_models/netG_A_encoder.onnx")
    netG_B_decoder = onnxruntime.InferenceSession("onnx_models/netG_B_decoder.onnx")
    mapping_net = onnxruntime.InferenceSession("onnx_models/mapping_net.onnx")

    return {
        "netG_A_encoder": netG_A_encoder,
        "netG_B_decoder": netG_B_decoder,
        "mapping_net": mapping_net
    }


def resize_image_tensor(img_tensor, target_size):
    t = transforms.Resize(target_size)
    img_tensor = t(torch.tensor(img_tensor))
    return img_tensor


def resize_if_large(rgb_tensor, mask_tensor, max_size=850):
    orig_size = None
    h, w = rgb_tensor.shape[-2], rgb_tensor.shape[-1]
    # 400 x 400 is an optimal size of pictures
    factor = int(max(h, w) // 300)

    if h > max_size or w > max_size:
        target_size = (h // factor, w // factor)
        rgb_tensor = resize_image_tensor(rgb_tensor, target_size)
        mask_tensor = resize_image_tensor(mask_tensor, target_size)
        rgb_tensor = to_numpy(rgb_tensor)
        mask_tensor = to_numpy(mask_tensor)
        orig_size = (h, w)
    return rgb_tensor, mask_tensor, orig_size


def run_model_parts(opt, sessions, inst, mask):
    netG_A_encoder = sessions["netG_A_encoder"]
    netG_B_decoder = sessions["netG_B_decoder"]
    mapping_net = sessions["mapping_net"]
    use_gpu = True if torch.cuda.is_available() else False

    if use_gpu:
        input_concat = mask.data.cuda()
        inst_data = inst.cuda()
    else:
        input_concat = mask.data
        inst_data = inst

    inst_data = to_numpy(inst_data)
    input_concat = to_numpy(input_concat)
    inst_data, input_concat, origin_size = resize_if_large(
        inst_data, input_concat)

    netG_A_encoder_inp = {
        netG_A_encoder.get_inputs()[0].name: to_numpy(inst_data),
    }

    netG_A_enc_out = netG_A_encoder.run(None, netG_A_encoder_inp)

    mapping_net_inp = {
        mapping_net.get_inputs()[0].name: netG_A_enc_out[0],
        mapping_net.get_inputs()[1].name: input_concat
    }

    mapping_net_out = mapping_net.run(None, mapping_net_inp)

    netG_B_decoder_inp = {
        netG_B_decoder.get_inputs()[0].name: to_numpy(mapping_net_out[0])}
    netG_B_dec_out = netG_B_decoder.run(None, netG_B_decoder_inp)

    if origin_size is not None:
        netG_B_dec_out = [resize_image_tensor(
            netG_B_dec_out[0], origin_size)]

    return netG_B_dec_out


if __name__ == "__main__":

    opt = TestOptions().parse(save=False)
    parameter_set(opt)

    model = Pix2PixHDModel_Mapping()
    if not os.path.exists("onnx_models"):
        model.initialize(opt)
        model.eval()
        os.makedirs("onnx_models", exist_ok=True)
        make_onnx_files(model)
    sessions = get_onnx_sessions()

    if not os.path.exists(opt.outputs_dir + "/" + "input_image"):
        os.makedirs(opt.outputs_dir + "/" + "input_image")
    if not os.path.exists(opt.outputs_dir + "/" + "restored_image"):
        os.makedirs(opt.outputs_dir + "/" + "restored_image")
    if not os.path.exists(opt.outputs_dir + "/" + "origin"):
        os.makedirs(opt.outputs_dir + "/" + "origin")

    dataset_size = 0

    os.makedirs(opt.test_input, exist_ok=True)
    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    if opt.test_mask != "":
        mask_loader = os.listdir(opt.test_mask)
        dataset_size = len(os.listdir(opt.test_mask))
        mask_loader.sort()

    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    for i in range(dataset_size):

        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")

        print("Now you are processing %s" % (input_name))

        if opt.NL_use_mask:
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
            if opt.mask_dilation != 0:
                kernel = np.ones((3,3),np.uint8)
                mask = np.array(mask)
                mask = cv2.dilate(mask,kernel,iterations = opt.mask_dilation)
                mask = Image.fromarray(mask.astype('uint8'))
            origin = input
            input = irregular_hole_synthesize(input, mask)
            mask = mask_transform(mask)
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            input = img_transform(input)
            input = input.unsqueeze(0)
        else:
            if opt.test_mode == "Scale":
                input = data_transforms(input, scale=True)
            if opt.test_mode == "Full":
                input = data_transforms(input, scale=False)
            if opt.test_mode == "Crop":
                input = data_transforms_rgb_old(input)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)
        ### Necessary input

        generated = run_model_parts(opt, sessions, input, mask)
        if input_name.endswith(".jpg"):
            input_name = input_name[:-4] + ".png"

        image_grid = vutils.save_image(
            (input + 1.0) / 2.0,
            opt.outputs_dir + "/input_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )

        image_grid = vutils.save_image(
            (torch.Tensor(generated[0]) + 1.0) / 2.0,
            opt.outputs_dir + "/restored_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )

        image_grid = ((torch.Tensor(generated[0]) + 1.0) / 2.0).numpy()
        image_grid = (255 * image_grid).astype(np.uint8)
        if len(image_grid.shape) > 3:
            image_grid = image_grid[0].transpose(1,2,0)
            image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            opt.outputs_dir + "/restored_image/" + input_name,
            image_grid)

        print(f"saved {input_name}")
        origin.save(opt.outputs_dir + "/origin/" + input_name)
