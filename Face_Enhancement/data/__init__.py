# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from data.face_dataset import FaceTestDataset


def create_dataloader(opt):

    instance = FaceTestDataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" % (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
    )
    return dataloader
