# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
import torchvision.utils as vutils



### " --tensorboard_log --no_instance --no_parsing_map" serial_batches, no_flip ###
def test_face(old_face_folder, old_face_label_folder, name, gpu_ids, load_size, label_nc, preprocess_mode, batchSize, results_dir,
 			checkpoints_dir="./checkpoints", how_many=50, tf_log = False, tensorboard_log = None, no_instance=False, no_parsing_map=False,
			serial_batches=False, nThreads=2, dataroot="./datasets/cityscapes/", isTrain=False, crop_size=512, aspect_ratio=1.0):

	dataloader = data.create_dataloader(batchSize, serial_batches, nThreads, isTrain, dataroot, old_face_folder, 
					old_face_label_folder, load_size, preprocess_mode, crop_size, no_flip, aspect_ratio)

	model = Pix2PixModel(opt)
	model.eval()

	visualizer = Visualizer(name=name, checkpoints_dir=checkpoints_dir, results_dir=results_dir, batchSize=batchSize,
							label_nc=label_nc, tf_log=tf_log, tensorboard_log=tensorboard_log)

	single_save_url = os.path.join(checkpoints_dir, name, results_dir, "each_img")


	if not os.path.exists(single_save_url):
	    os.makedirs(single_save_url)


	for i, data_i in enumerate(dataloader):
	    if i * batchSize >= how_many:
	        break

	    generated = model(data_i, mode="inference")

	    img_path = data_i["path"]

	    for b in range(generated.shape[0]):
	        img_name = os.path.split(img_path[b])[-1]
	        save_img_url = os.path.join(single_save_url, img_name)

	        vutils.save_image((generated[b] + 1) / 2, save_img_url)

