# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import torch


class FaceTestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--no_pairing_check",
            action="store_true",
            help="If specified, skip sanity check of correct label-image file pairing",
        )
        #    parser.set_defaults(contain_dontcare_label=False)
        #    parser.set_defaults(no_instance=True)
        return parser

    def initialize(self, dataroot, old_face_folder, old_face_label_folder, load_size, preprocess_mode, 
                    crop_size, no_flip, aspect_ratio):
        # self.opt = opt
        self.aspect_ratio = aspect_ratio
        self.no_flip = no_flip
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.dataroot = dataroot
        self.load_size = load_size
        self.old_face_folder = old_face_folder

        image_path = os.path.join(dataroot, old_face_folder)
        label_path = os.path.join(dataroot, old_face_label_folder)

        image_list = os.listdir(image_path)
        image_list = sorted(image_list)
        # image_list=image_list[:opt.max_dataset_size]

        self.label_paths = label_path  ## Just the root dir
        self.image_paths = image_list  ## All the image name

        self.parts = [
            "skin",
            "hair",
            "l_brow",
            "r_brow",
            "l_eye",
            "r_eye",
            "eye_g",
            "l_ear",
            "r_ear",
            "ear_r",
            "nose",
            "mouth",
            "u_lip",
            "l_lip",
            "neck",
            "neck_l",
            "cloth",
            "hat",
        ]

        size = len(self.image_paths)
        self.dataset_size = size

    def __getitem__(self, index):

        # get_params(preprocess_mode, load_size, crop_size, size)
        params = get_params(self.preprocess_mode, self.load_size, self.crop_size, (-1, -1))
        image_name = self.image_paths[index]
        image_path = os.path.join(self.dataroot, self.old_face_folder, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")

        transform_image = get_transform(self.preprocess_mode, self.load_size, self.crop_size, 
                                        self.aspect_ratio, self.no_flip, self.isTrain, params)
        image_tensor = transform_image(image)

        img_name = image_name[:-4]

        transform_label = get_transform(self.preprocess_mode, self.load_size, self.crop_size, self.aspect_ratio,
                                     self.no_flip, self.isTrain, params, method=Image.NEAREST, normalize=False)
        full_label = []

        cnt = 0

        for each_part in self.parts:
            part_name = img_name + "_" + each_part + ".png"
            part_url = os.path.join(self.label_paths, part_name)

            if os.path.exists(part_url):
                label = Image.open(part_url).convert("RGB")
                label_tensor = transform_label(label)  ## 3 channels and pixel [0,1]
                full_label.append(label_tensor[0])
            else:
                current_part = torch.zeros((self.load_size, self.load_size))
                full_label.append(current_part)
                cnt += 1

        full_label_tensor = torch.stack(full_label, 0)

        input_dict = {
            "label": full_label_tensor,
            "image": image_tensor,
            "path": image_path,
        }

        return input_dict

    def __len__(self):
        return self.dataset_size

