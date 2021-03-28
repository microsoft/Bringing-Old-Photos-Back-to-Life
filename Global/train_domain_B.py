# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import random


opt = TrainOptions().parse()

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset) * opt.batchSize
print('#training images = %d' % dataset_size)

path = os.path.join(opt.checkpoints_dir, opt.name, 'model.txt')
visualizer = Visualizer(opt)


iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    visualizer.print_save('Resuming from epoch %d at iteration %d' % (start_epoch-1, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# opt.which_epoch=start_epoch-1
model = create_model(opt)
fd = open(path, 'w')
fd.write(str(model.module.netG))
fd.write(str(model.module.netD))
fd.close()

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))


        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict['G_KL'] + loss_dict.get('Smooth_L1',0)*opt.L1_weight


        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, model.module.old_lr)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:

            if not os.path.exists(opt.outputs_dir + opt.name):
                os.makedirs(opt.outputs_dir + opt.name)
            imgs_num = 5
            imgs = torch.cat((data['label'][:imgs_num], generated.data.cpu()[:imgs_num], data['image'][:imgs_num]), 0)

            imgs = (imgs + 1.) / 2.0

            try:
                image_grid = vutils.save_image(imgs, opt.outputs_dir + opt.name + '/' + str(epoch) + '_' + str(total_steps) + '.png',
                        nrow=imgs_num, padding=0, normalize=True)
            except OSError as err:
                print(err)

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

