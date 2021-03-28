# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def create_model(opt):
    if opt.model == "pix2pixHD":
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel

        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel

        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) > 1:
        # pass
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model

def create_da_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model_DA import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) > 1:
        #pass
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model