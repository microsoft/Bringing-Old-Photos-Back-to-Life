# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, g_kl, d_real, d_fake, g_featd, featd_real, featd_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_kl, d_real, d_fake, g_featd, featd_real, featd_fake), flags) if f]

        return loss_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat  ## Clearly it is false
        self.gen_features = self.use_features and not self.opt.load_features  ## it is also false
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  ## Just is the origin input channel #

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, opt.k_size,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids, opt=opt)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc if opt.no_cgan else input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt,opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

            self.feat_D=networks.define_D(64, opt.ndf, opt.n_layers_D, opt, opt.norm, use_sigmoid,
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)

            print("---------- G Networks reloaded -------------")
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.feat_D, 'feat_D', opt.which_epoch, pretrained_path)
                print("---------- D Networks reloaded -------------")


                # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:  ## The pool_size is 0!
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss_torch(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'G_KL', 'D_real', 'D_fake', 'G_featD', 'featD_real','featD_fake')

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.feat_D.parameters())
            self.optimizer_featD = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            print("---------- Optimizers initialized -------------")

            if opt.continue_train:
                self.load_optimizer(self.optimizer_D, 'D', opt.which_epoch)
                self.load_optimizer(self.optimizer_G, "G", opt.which_epoch)
                self.load_optimizer(self.optimizer_featD,'featD',opt.which_epoch)
                for param_groups in self.optimizer_D.param_groups:
                    self.old_lr = param_groups['lr']

                print("---------- Optimizers reloaded -------------")
                print("---------- Current LR is %.8f -------------" % (self.old_lr))

            ## We also want to re-load the parameters of optimizer.

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        if input_label is None:
            input_concat = test_image.detach()
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def feat_discriminate(self,input):

        return self.feat_D.forward(input.detach())


    def forward(self, label, inst, image, feat, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        hiddens = self.netG.forward(input_concat, 'enc')
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        # We follow the the VAE of MUNIT (https://github.com/NVlabs/MUNIT/blob/master/networks.py)
        fake_image = self.netG.forward(hiddens + noise, 'dec')

        ####################
        ##### GAN for the intermediate feature
        real_old_feat =[]
        syn_feat = []
        for index,x in enumerate(inst):
            if x==1:
                real_old_feat.append(hiddens[index].unsqueeze(0))
            else:
                syn_feat.append(hiddens[index].unsqueeze(0))
        L=min(len(real_old_feat),len(syn_feat))
        real_old_feat=real_old_feat[:L]
        syn_feat=syn_feat[:L]
        real_old_feat=torch.cat(real_old_feat,0)
        syn_feat=torch.cat(syn_feat,0)

        pred_fake_feat=self.feat_discriminate(real_old_feat)
        loss_featD_fake = self.criterionGAN(pred_fake_feat, False)
        pred_real_feat=self.feat_discriminate(syn_feat)
        loss_featD_real = self.criterionGAN(pred_real_feat, True)

        pred_fake_feat_G=self.feat_D.forward(real_old_feat)
        loss_G_featD=self.criterionGAN(pred_fake_feat_G,True)


        #####################################
        if self.opt.no_cgan:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(None, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = self.discriminate(None, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = self.netD.forward(fake_image)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = self.discriminate(input_label, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)

        loss_G_kl = torch.mean(torch.pow(hiddens, 2)) * self.opt.kl

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i]) - 1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                                       self.criterionFeat(pred_fake[i][j],
                                                          pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_kl, loss_D_real, loss_D_fake,loss_G_featD, loss_featD_real, loss_featD_fake),
                None if not infer else fake_image]

    def inference(self, label, inst, image=None, feat=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num // 2, :]
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.feat_D,'featD',which_epoch,self.gpu_ids)

        self.save_optimizer(self.optimizer_G, "G", which_epoch)
        self.save_optimizer(self.optimizer_D, "D", which_epoch)
        self.save_optimizer(self.optimizer_featD,'featD',which_epoch)

        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):

        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_featD.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
