# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--ntest", type=int, default=float("inf"), help="# of test examples.")
        self.parser.add_argument("--results_dir", type=str, default="./results/", help="saves results here.")
        self.parser.add_argument(
            "--aspect_ratio", type=float, default=1.0, help="aspect ratio of result images"
        )
        self.parser.add_argument("--phase", type=str, default="test", help="train, val, test, etc")
        self.parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        self.parser.add_argument("--how_many", type=int, default=50, help="how many test images to run")
        self.parser.add_argument(
            "--cluster_path",
            type=str,
            default="features_clustered_010.npy",
            help="the path for clustered results of encoded features",
        )
        self.parser.add_argument(
            "--use_encoded_image",
            action="store_true",
            help="if specified, encode the real image to get the feature map",
        )
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
        self.parser.add_argument(
            "--start_epoch",
            type=int,
            default=-1,
            help="write the start_epoch of iter.txt into this parameter",
        )

        self.parser.add_argument("--test_dataset", type=str, default="Real_RGB_old.bigfile")
        self.parser.add_argument(
            "--no_degradation",
            action="store_true",
            help="when train the mapping, enable this parameter --> no degradation will be added into clean image",
        )
        self.parser.add_argument(
            "--no_load_VAE",
            action="store_true",
            help="when train the mapping, enable this parameter --> random initialize the encoder an decoder",
        )
        self.parser.add_argument(
            "--use_v2_degradation",
            action="store_true",
            help="enable this parameter --> 4 kinds of degradations will be used to synthesize corruption",
        )
        self.parser.add_argument("--use_vae_which_epoch", type=str, default="latest")
        self.isTrain = False

        self.parser.add_argument("--generate_pair", action="store_true")

        self.parser.add_argument("--multi_scale_test", type=float, default=0.5)
        self.parser.add_argument("--multi_scale_threshold", type=float, default=0.5)
        self.parser.add_argument(
            "--mask_need_scale",
            action="store_true",
            help="enable this param meas that the pixel range of mask is 0-255",
        )
        self.parser.add_argument("--scale_num", type=int, default=1)

        self.parser.add_argument(
            "--save_feature_url", type=str, default="", help="While extracting the features, where to put"
        )

        self.parser.add_argument(
            "--test_input", type=str, default="", help="A directory or a root of bigfile"
        )
        self.parser.add_argument("--test_mask", type=str, default="", help="A directory or a root of bigfile")
        self.parser.add_argument("--test_gt", type=str, default="", help="A directory or a root of bigfile")

        self.parser.add_argument(
            "--scale_input", action="store_true", help="While testing, choose to scale the input firstly"
        )

        self.parser.add_argument(
            "--save_feature_name", type=str, default="features.json", help="The name of saved features"
        )
        self.parser.add_argument(
            "--test_rgb_old_wo_scratch", action="store_true", help="Same setting with origin test"
        )

        self.parser.add_argument("--test_mode", type=str, default="Crop", help="Scale|Full|Crop")
        self.parser.add_argument("--Quality_restore", action="store_true", help="For RGB images")
        self.parser.add_argument(
            "--Scratch_and_Quality_restore", action="store_true", help="For scratched images"
        )
        self.parser.add_argument("--HR", action='store_true',help='Large input size with scratches')
