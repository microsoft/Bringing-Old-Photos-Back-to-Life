# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import ntpath
import time
from . import util
import scipy.misc

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch
import numpy as np


class Visualizer:
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log

        self.tensorboard_log = opt.tensorboard_log

        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.tensorboard_log:

            if self.opt.isTrain:
                self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writer = SummaryWriter(log_dir=self.log_dir)
            else:
                print("hi :)")
                self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.results_dir)
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)

        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, "loss_log.txt")
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write("================ Training Loss (%s) ================\n" % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        all_tensor = []
        if self.tensorboard_log:

            for key, tensor in visuals.items():
                all_tensor.append((tensor.data.cpu() + 1) / 2)

            output = torch.cat(all_tensor, 0)
            img_grid = vutils.make_grid(output, nrow=self.opt.batchSize, padding=0, normalize=False)

            if self.opt.isTrain:
                self.writer.add_image("Face_SPADE/training_samples", img_grid, step)
            else:
                vutils.save_image(
                    output,
                    os.path.join(self.log_dir, str(step) + ".png"),
                    nrow=self.opt.batchSize,
                    padding=0,
                    normalize=False,
                )

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

        if self.tensorboard_log:

            self.writer.add_scalar("Loss/GAN_Feat", errors["GAN_Feat"].mean().float(), step)
            self.writer.add_scalar("Loss/VGG", errors["VGG"].mean().float(), step)
            self.writer.add_scalars(
                "Loss/GAN",
                {
                    "G": errors["GAN"].mean().float(),
                    "D": (errors["D_Fake"].mean().float() + errors["D_real"].mean().float()) / 2,
                },
                step,
            )

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, i, t)
        for k, v in errors.items():
            v = v.mean().float()
            message += "%s: %.3f " % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batchSize > 8
            if "input_label" == key:
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)  ## B*H*W*C 0-255 numpy
            else:
                t = util.tensor2im(t, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, "%s.png" % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
