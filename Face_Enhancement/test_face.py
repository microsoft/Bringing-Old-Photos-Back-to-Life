# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict

import data
import numpy as np
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
import torch
import torchvision.utils as vutils
import onnxruntime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

opt = TestOptions().parse()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs("onnx_models", exist_ok=True)

if not os.path.exists("onnx_models/enchancment_netG.onnx"):
    model = Pix2PixModel(opt)
    model.eval()

    dummy_inputs = (
        torch.randn(1, 18, 512, 512, requires_grad=True, device=device),
        torch.randn(1, 3, 512, 512, requires_grad=True, device=device)
    )
    torch.onnx.export(
        model.netG,
        dummy_inputs,
        "onnx_models/enchancment_netG.onnx",
        opset_version=11,
        input_names = ['input_semantics', 'degraded_image'],
        output_names = ['output'],
        dynamic_axes = {
            'input_semantics': [0, 2, 3],
            'degraded_image': [0, 2, 3],
            'output': [0, 2, 3]
        }
    )

session = onnxruntime.InferenceSession("onnx_models/enchancment_netG.onnx")


def to_numpy(tensor):
    if isinstance(tensor, list):
        return np.array(tensor)
    elif isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess_input(data):
    data["label"] = data["label"].to(device)
    data["image"] = data["image"].to(device)
    return data["label"], data["image"]


def generate_fake(data):
    input_semantics, real_image = preprocess_input(data)
    input = {
        session.get_inputs()[0].name: to_numpy(real_image)
    }

    out = session.run(None, input)
    return out[0]

dataloader = data.create_dataloader(opt)
visualizer = Visualizer(opt)
single_save_url = os.path.join(
    opt.checkpoints_dir, opt.name, opt.results_dir, "each_img")


if not os.path.exists(single_save_url):
    os.makedirs(single_save_url)


for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = generate_fake(data_i)
    img_path = data_i["path"]

    for b in range(generated.shape[0]):
        img_name = os.path.split(img_path[b])[-1]
        save_img_url = os.path.join(single_save_url, img_name)

        vutils.save_image((torch.from_numpy(generated[b]) + 1) / 2, save_img_url)
