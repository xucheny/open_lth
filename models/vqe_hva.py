# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    '''A torch model for Hamiltonian variational ansatz'''

    def __init__(self, parameterized_ops, observable, initializer):
        super(Model, self).__init__()

        self.ansatz = Ansatz_HVA(parameterized_ops, num_layers) None # ansatz
        self.energy_module = Energy(observable) None # including the g
        self.criterion = Placeholderloss() # identity, does not use the other input

        self.apply(initializer)

    def forward(self, x):
        '''
        the data will not be used at all
        '''
        output_state = self.ansatz()
        energy = self.energy_module(output_state)
        return energy # what will be the shape?

    @property
    def output_layer_names(self):
        raise NotImplementedError('output_layer_names not implemented for: vqe_hva')

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('mnist_lenet') and
                len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]))

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is vqe_hva_HVANAME_NLAYERS_size[NSIZE1_NSIZE2...]_weight[_w2...].
        HVANAME is the name of the module
        NSIZE is the description of system size
        NLAYER is the number of layers
        w1, w2, etc. are the reweighting factor of components in the hamiltonians
        For example, a 3-qubit TFIM1d model with observable ZZ + 0.1 X is 'vqe_hva_TFIM1d_10
        A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        outputs = outputs or 1

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        HVANAME = model_name.split('_')[2]
        if HVANAME == 'TFIM1d':
            pass
        elif HVANAME == 'XXZ1d':
            pass



        plan = [int(n) for n in model_name.split('_')[2:]]
        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_lenet_300_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='placeholder',
            batch_size=1
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=0.1,
            weight_decay=1e-4,
            training_steps='1000ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
