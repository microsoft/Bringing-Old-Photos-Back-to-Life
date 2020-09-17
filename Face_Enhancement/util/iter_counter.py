# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import numpy as np


# Helper class that keeps track of training iterations
class IterationCounter:
    def __init__(self, opt, dataset_size):
        self.opt = opt
        self.dataset_size = dataset_size

        self.first_epoch = 1
        self.total_epochs = opt.niter + opt.niter_decay
        self.epoch_iter = 0  # iter number within each epoch
        self.iter_record_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "iter.txt")
        if opt.isTrain and opt.continue_train:
            try:
                self.first_epoch, self.epoch_iter = np.loadtxt(
                    self.iter_record_path, delimiter=",", dtype=int
                )
                print("Resuming from epoch %d at iteration %d" % (self.first_epoch, self.epoch_iter))
            except:
                print(
                    "Could not load iteration record at %s. Starting from beginning." % self.iter_record_path
                )

        self.total_steps_so_far = (self.first_epoch - 1) * dataset_size + self.epoch_iter

    # return the iterator of epochs for the training
    def training_epochs(self):
        return range(self.first_epoch, self.total_epochs + 1)

    def record_epoch_start(self, epoch):
        self.epoch_start_time = time.time()
        self.epoch_iter = 0
        self.last_iter_time = time.time()
        self.current_epoch = epoch

    def record_one_iteration(self):
        current_time = time.time()

        # the last remaining batch is dropped (see data/__init__.py),
        # so we can assume batch size is always opt.batchSize
        self.time_per_iter = (current_time - self.last_iter_time) / self.opt.batchSize
        self.last_iter_time = current_time
        self.total_steps_so_far += self.opt.batchSize
        self.epoch_iter += self.opt.batchSize

    def record_epoch_end(self):
        current_time = time.time()
        self.time_per_epoch = current_time - self.epoch_start_time
        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (self.current_epoch, self.total_epochs, self.time_per_epoch)
        )
        if self.current_epoch % self.opt.save_epoch_freq == 0:
            np.savetxt(self.iter_record_path, (self.current_epoch + 1, 0), delimiter=",", fmt="%d")
            print("Saved current iteration count at %s." % self.iter_record_path)

    def record_current_iter(self):
        np.savetxt(self.iter_record_path, (self.current_epoch, self.epoch_iter), delimiter=",", fmt="%d")
        print("Saved current iteration count at %s." % self.iter_record_path)

    def needs_saving(self):
        return (self.total_steps_so_far % self.opt.save_latest_freq) < self.opt.batchSize

    def needs_printing(self):
        return (self.total_steps_so_far % self.opt.print_freq) < self.opt.batchSize

    def needs_displaying(self):
        return (self.total_steps_so_far % self.opt.display_freq) < self.opt.batchSize
