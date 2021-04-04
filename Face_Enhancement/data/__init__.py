# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.face_dataset import FaceTestDataset

# dataroot, old_face_folder, old_face_label_folder, load_size, preprocess_mode, 
#                     crop_size, no_flip, aspect_ratio

def create_dataloader(batchSize, serial_batches, nThreads, isTrain, dataroot, old_face_folder, old_face_label_folder, 
                load_size, preprocess_mode, crop_size, no_flip, aspect_ratio):

    instance = FaceTestDataset()
    instance.initialize(dataroot, old_face_folder, old_face_label_folder, load_size, preprocess_mode, 
                    crop_size, no_flip, aspect_ratio)
    print("dataset [%s] of size %d was created" % (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=batchSize,
        shuffle=not serial_batches,
        num_workers=int(nThreads),
        drop_last=isTrain,
    )
    return dataloader
