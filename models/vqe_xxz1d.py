# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global, parallel_sparse_global

from utils.qc_model_utils import Energy, PlaceholderLoss, HamiltonianVariationalAnsatz, get_xxz1d_setup


class Model(base.Model):
    '''A torch model for Hamiltonian variational ansatz'''

    def __init__(self, num_qubits, num_layers, delta, initializer):
        print('Model: vqe_xxz1d')
        super(Model, self).__init__()
        parameterized_ops, observable, input_state = get_xxz1d_setup(num_qubits, delta)
        self.ansatz = HamiltonianVariationalAnsatz(parameterized_ops, num_layers, input_state, 100)
        self.energy_module = Energy(observable)
        self.criterion = PlaceholderLoss() # simply sum or mean, does not use the other input

        self.apply(initializer)

    def forward(self, x):
        '''
        the data will not be used at all
        '''
        output_state = self.ansatz()
        energy = self.energy_module(output_state)
        return energy # what will be the shape?

    @property
    def prunable_layer_names(self):
        return [name + '.params' for name, module in self.named_modules() if
                isinstance(module, HamiltonianVariationalAnsatz)]

    @property
    def output_layer_names(self):
        return []
        raise NotImplementedError('output_layer_names not implemented for: vqe_hva')

    @staticmethod
    def is_valid_model_name(model_name):
        def isfloat(value:str):
            try:
                float(value)
                return True
            except ValueError:
                return False

        return (
                model_name.startswith('vqe_xxz1d') and
                len(model_name.split('_')) == 5 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:4]]) and
                all([isfloat(x) and float(x) > 0 for x in model_name.split('_')[4:]])
                )

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """The name of a model is vqe_xxz1d_Nqubits_Nlayers_delta
        Nqubits is the num of qubits in the system,
        Nlayers is the number of layers of the HVA,
        delta is the magnitude of the ZZ term,
        For example, a 4-qubit 10 layer xxz1d model,
        with observable XX + YY + 0.5 * ZZ is 'vqe_xxz1d_4_10_0.5'.
        """

        outputs = outputs or 1

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        num_qubits = int(model_name.split('_')[2])
        num_layers = int(model_name.split('_')[3])
        delta = float(model_name.split('_')[4])

        return Model(num_qubits, num_layers, delta, initializer)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='vqe_xxz1d_8_100_0.9',
            model_init='hva_normal',
            batchnorm_init='uniform' # Does not matter, we don't have module named 'BatchNorm2d'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='placeholder',
            batch_size=3
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='adam',
            lr=1e-3,
            weight_decay=1e-4,
            training_steps='3000ep',
        )

        pruning_hparams = parallel_sparse_global.PruningHparams(
            pruning_strategy='parallel_sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore=None #TODO: what if we have nothing to ignore?
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
