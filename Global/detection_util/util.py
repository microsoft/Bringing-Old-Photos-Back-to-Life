# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import time
import shutil
import platform
import numpy as np
from datetime import datetime

import torch
import torchvision as tv
import torch.backends.cudnn as cudnn

# from torch.utils.tensorboard import SummaryWriter

import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import torchvision.utils as vutils


##### option parsing ######
def print_options(config_dict):
    print("------------ Options -------------")
    for k, v in sorted(config_dict.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------- End ----------------")


def save_options(config_dict):
    from time import gmtime, strftime

    file_dir = os.path.join(config_dict["checkpoint_dir"], config_dict["name"])
    mkdir_if_not(file_dir)
    file_name = os.path.join(file_dir, "opt.txt")
    with open(file_name, "wt") as opt_file:
        opt_file.write(os.path.basename(sys.argv[0]) + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
        opt_file.write("------------ Options -------------\n")
        for k, v in sorted(config_dict.items()):
            opt_file.write("%s: %s\n" % (str(k), str(v)))
        opt_file.write("-------------- End ----------------\n")


def config_parse(config_file, options, save=True):
    with open(config_file, "r") as stream:
        config_dict = yaml.safe_load(stream)
        config = edict(config_dict)

    for option_key, option_value in vars(options).items():
        config_dict[option_key] = option_value
        config[option_key] = option_value

    if config.debug_mode:
        config_dict["num_workers"] = 0
        config.num_workers = 0
        config.batch_size = 2
        if isinstance(config.gpu_ids, str):
            config.gpu_ids = [int(x) for x in config.gpu_ids.split(",")][0]

    print_options(config_dict)
    if save:
        save_options(config_dict)

    return config


###### utility ######
def to_np(x):
    return x.cpu().numpy()


def prepare_device(use_gpu, gpu_ids):
    if use_gpu:
        cudnn.benchmark = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if isinstance(gpu_ids, str):
            gpu_ids = [int(x) for x in gpu_ids.split(",")]
            torch.cuda.set_device(gpu_ids[0])
            device = torch.device("cuda:" + str(gpu_ids[0]))
        else:
            torch.cuda.set_device(gpu_ids)
            device = torch.device("cuda:" + str(gpu_ids))
        print("running on GPU {}".format(gpu_ids))
    else:
        device = torch.device("cpu")
        print("running on CPU")

    return device


###### file system ######
def get_dir_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def mkdir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


##### System related ######
class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        elapse = time.time() - self.start_time
        print(self.msg % elapse)


###### interactive ######
def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def clean_tensorboard(directory):
    tensorboard_list = os.listdir(directory)
    SIZE_THRESH = 100000
    for tensorboard in tensorboard_list:
        tensorboard = os.path.join(directory, tensorboard)
        if get_size(tensorboard) < SIZE_THRESH:
            print("deleting the empty tensorboard: ", tensorboard)
            #
            if os.path.isdir(tensorboard):
                shutil.rmtree(tensorboard)
            else:
                os.remove(tensorboard)


def prepare_tensorboard(config, experiment_name=datetime.now().strftime("%Y-%m-%d %H-%M-%S")):
    tensorboard_directory = os.path.join(config.checkpoint_dir, config.name, "tensorboard_logs")
    mkdir_if_not(tensorboard_directory)
    clean_tensorboard(tensorboard_directory)
    tb_writer = SummaryWriter(os.path.join(tensorboard_directory, experiment_name), flush_secs=10)

    # try:
    #     shutil.copy('outputs/opt.txt', tensorboard_directory)
    # except:
    #     print('cannot find file opt.txt')
    return tb_writer


def tb_loss_logger(tb_writer, iter_index, loss_logger):
    for tag, value in loss_logger.items():
        tb_writer.add_scalar(tag, scalar_value=value.item(), global_step=iter_index)


def tb_image_logger(tb_writer, iter_index, images_info, config):
    ### Save and write the output into the tensorboard
    tb_logger_path = os.path.join(config.output_dir, config.name, config.train_mode)
    mkdir_if_not(tb_logger_path)
    for tag, image in images_info.items():
        if tag == "test_image_prediction" or tag == "image_prediction":
            continue
        image = tv.utils.make_grid(image.cpu())
        image = torch.clamp(image, 0, 1)
        tb_writer.add_image(tag, img_tensor=image, global_step=iter_index)
        tv.transforms.functional.to_pil_image(image).save(
            os.path.join(tb_logger_path, "{:06d}_{}.jpg".format(iter_index, tag))
        )


def tb_image_logger_test(epoch, iter, images_info, config):

    url = os.path.join(config.output_dir, config.name, config.train_mode, "val_" + str(epoch))
    if not os.path.exists(url):
        os.makedirs(url)
    scratch_img = images_info["test_scratch_image"].data.cpu()
    if config.norm_input:
        scratch_img = (scratch_img + 1.0) / 2.0
    scratch_img = torch.clamp(scratch_img, 0, 1)
    gt_mask = images_info["test_mask_image"].data.cpu()
    predict_mask = images_info["test_scratch_prediction"].data.cpu()

    predict_hard_mask = (predict_mask.data.cpu() >= 0.5).float()

    imgs = torch.cat((scratch_img, predict_hard_mask, gt_mask), 0)
    img_grid = vutils.save_image(
        imgs, os.path.join(url, str(iter) + ".jpg"), nrow=len(scratch_img), padding=0, normalize=True
    )


def imshow(input_image, title=None, to_numpy=False):
    inp = input_image
    if to_numpy or type(input_image) is torch.Tensor:
        inp = input_image.numpy()

    fig = plt.figure()
    if inp.ndim == 2:
        fig = plt.imshow(inp, cmap="gray", clim=[0, 255])
    else:
        fig = plt.imshow(np.transpose(inp, [1, 2, 0]).astype(np.uint8))
    plt.axis("off")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


###### vgg preprocessing ######
def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(
        1, 3, 1, 1
    )
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst


def torch_vgg_preprocess(tensor):
    # pytorch version normalization
    # note that both input and output are RGB tensors;
    # input and output ranges in [0,1]
    # normalize the tensor with mean and variance
    tensor_mc = tensor - torch.Tensor([0.485, 0.456, 0.406]).type_as(tensor).view(1, 3, 1, 1)
    tensor_mc_norm = tensor_mc / torch.Tensor([0.229, 0.224, 0.225]).type_as(tensor_mc).view(1, 3, 1, 1)
    return tensor_mc_norm


def network_gradient(net, gradient_on=True):
    if gradient_on:
        for param in net.parameters():
            param.requires_grad = True
    else:
        for param in net.parameters():
            param.requires_grad = False
    return net
