# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument("--results_dir", type=str, default="./results/", help="saves results here.")
        parser.add_argument(
            "--which_epoch",
            type=str,
            default="latest",
            help="which epoch to load? set to latest to use latest cached model",
        )
        parser.add_argument("--how_many", type=int, default=float("inf"), help="how many test images to run")

        parser.set_defaults(
            preprocess_mode="scale_width_and_crop", crop_size=256, load_size=256, display_winsize=256
        )
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase="test")
        self.isTrain = False
        return parser
