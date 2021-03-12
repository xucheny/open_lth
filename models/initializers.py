# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from utils import qc_model_utils
import functools

def binary(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
        sigma = w.weight.data.std()
        w.weight.data = torch.sign(w.weight.data) * sigma


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)


def kaiming_uniform(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)


def orthogonal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(w.weight)

##########################################################################
def hva_normal(w, std=1e-1):
    if isinstance(w, qc_model_utils.HamiltonianVariationalAnsatz):
        torch.nn.init.normal_(w.params, mean=0, std=std)
def hva_normal_1(w):
    hva_normal(w, 1)
def hva_normal_10(w):
    hva_normal(w, 10)

def hva_uniform(w, bound=10):
    pass
def hva_zeros(w):
    if isinstance(w, qc_model_utils.HamiltonianVariationalAnsatz):
        torch.nn.init.zeros_(w.params)
