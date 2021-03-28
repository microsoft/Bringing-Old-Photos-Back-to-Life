# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import sys


class BaseModel(torch.nn.Module):
    def name(self):
        return "BaseModel"

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = "%s_optimizer_%s.pth" % (epoch_label, optimizer_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, optimizer_label, epoch_label, save_dir=""):
        save_filename = "%s_optimizer_%s.pth" % (epoch_label, optimizer_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)

        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
        else:
            optimizer.load_state_dict(torch.load(save_path))

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=""):
        save_filename = "%s_net_%s.pth" % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir

        # print(save_dir)
        # print(self.save_dir)
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            # if network_label == 'G':
            #     raise('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                # print(save_path)
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    # if self.opt.verbose:
                    print(
                        "Pretrained network %s has excessive layers; Only loading layers that are used"
                        % network_label
                    )
                except:
                    print(
                        "Pretrained network %s has fewer layers; The following are not initialized:"
                        % network_label
                    )
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set

                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split(".")[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def update_learning_rate():
        pass
