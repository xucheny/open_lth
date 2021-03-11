# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from PIL import Image
import torchvision

from datasets import base
from platforms.platform import get_platform

import numpy as np

class Dataset(base.Dataset):
    """The MNIST dataset."""

    @staticmethod
    def num_train_examples(): return 3

    @staticmethod
    def num_test_examples(): return 3

    @staticmethod
    def num_classes(): return 1

    @staticmethod
    def get_train_set(use_augmentation):
        # a shape of 3 x 3 as placeholder
        return Dataset(
                np.ones((3,1)).astype('uint8'),
                np.zeros((3,)).astype('int64')
                )

    @staticmethod
    def get_test_set():
        return Dataset(
                np.ones((3,1)).astype('uint8'),
                np.zeros((3,)).astype('int64')
                )

    def __init__(self,  examples, labels):
        super(Dataset, self).__init__(examples, labels)

DataLoader = base.DataLoader
