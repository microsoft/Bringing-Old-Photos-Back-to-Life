# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        # self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--training_dataset',type=str,default='',help='training use which dataset')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--l2_feat', type=float, help='weight for feature mapping loss') 
        self.parser.add_argument('--use_l1_feat', action='store_true', help='use l1 for feat mapping')               
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--gan_type', type=str, default='lsgan', help='Choose the loss type of GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--norm_D',type=str, default='spectralinstance', help='instance normalization or batch normalization')
        self.parser.add_argument('--init_D',type=str,default='xavier',help='normal|xavier|xavier_uniform|kaiming|orthogonal|none')

        self.parser.add_argument('--no_TTUR',action='store_true',help='No TTUR')

        self.parser.add_argument('--start_epoch',type=int,default=-1,help='write the start_epoch of iter.txt into this parameter')
        self.parser.add_argument('--no_degradation',action='store_true',help='when train the mapping, enable this parameter --> no degradation will be added into clean image')
        self.parser.add_argument('--no_load_VAE',action='store_true',help='when train the mapping, enable this parameter --> random initialize the encoder an decoder')
        self.parser.add_argument('--use_v2_degradation',action='store_true',help='enable this parameter --> 4 kinds of degradations will be used to synthesize corruption')
        self.parser.add_argument('--use_vae_which_epoch',type=str,default='200')


        self.parser.add_argument('--use_focal_loss',action='store_true')

        self.parser.add_argument('--mask_need_scale',action='store_true',help='enable this param means that the pixel range of mask is 0-255')
        self.parser.add_argument('--positive_weight',type=float,default=1.0,help='(For scratch detection) Since the scratch number is less, and we use a weight strategy. This parameter means that we want to decrease the weight.')

        self.parser.add_argument('--no_update_lr',action='store_true',help='use this means we do not update the LR while training')


        self.isTrain = True
