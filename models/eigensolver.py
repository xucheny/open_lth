# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global

import numpy as np
from utils.qc_model_utils import PlaceholderLoss, get_xxz1d_setup, get_tfim1d_setup


class Model(base.Model):
    '''A torch model for Hamiltonian variational ansatz'''

    def __init__(self, plan, num_qubits, problem_name, delta, initializer):
        super(Model, self).__init__()
        if problem_name == 'tfim1d':
            _, observable, input_state = get_tfim1d_setup(num_qubits, delta)
        #elif problem_name == 'xxz1d':
        #    _, observable, input_state = get_xxz1d_setup(num_qubits, delta)

        else:
            raise NotImplementedError('Problem hamiltonian not implemented.')
        dim = 2 ** num_qubits

        layers = []
        current_size = dim  # input_size is the same of number of qubits
        for size in plan:
            layers.append(nn.Linear(current_size, size))
            current_size = size

        self.fc_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(current_size, dim)
        self.criterion = PlaceholderLoss() # simply sum or mean, does not use the other input

        # observable and input states must be real
        self.register_buffer('observable', torch.tensor(observable.real, dtype=torch.float))
        self.register_buffer('input_state', torch.tensor(input_state.real.reshape(-1,), dtype=torch.float))
        vals, vecs = np.linalg.eigh(observable)
        self.optimal_value =  vals[0]
        print('minimum val: {}'.format(vals[0]))


        self.apply(initializer)

    def forward(self, x):
        '''
        the data x will not be used at all
        '''
        state = self.input_state
        for layer in self.fc_layers:
            state = F.relu(layer(state))
        state = self.fc(state)

        state = state / torch.norm(state)
        energy = torch.matmul(state.reshape(1,-1), torch.matmul(self.observable, state.reshape(-1,1))) - self.optimal_value
        return energy # what will be the shape?

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    #@property
    #def prunable_layer_names(self):
    #    return [name + '.params' for name, module in self.named_modules() if
    #            isinstance(module, HamiltonianVariationalAnsatz)]

    @staticmethod
    def is_valid_model_name(model_name):
        def isfloat(value:str):
            try:
                float(value)
                return True
            except ValueError:
                return False

        def isinstanceinfo(value:str):
            '''
            example: xxz1d-8-1.0
            '''
            problem_name, num_qubits, delta = value.split('-')
            return (
                    problem_name in ['xxz1d', 'tfim1d'] and
                    num_qubits.isdigit() and int(num_qubits) > 0 and
                    isfloat(delta)
                    )

        return (model_name.startswith('eigensolver_') and
                len(model_name.split('_')) > 2 and
                isinstanceinfo(model_name.split('_')[1]) and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]])
                )

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=None):
        """
        Nqubits is the num of qubits in the system,
        delta is the magnitude of the ZZ term,
        For example, a 4-qubit 10 layer xxz1d model,
        with observable XX + YY + 1.0 * ZZ is

        model_name='eigensolver_xxz1d-8-1.0_30_10',
        """

        outputs = outputs or 1

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        instance_info = model_name.split('_')[1]
        plan = [int(n) for n in model_name.split('_')[2:]]

        problem_name = instance_info.split('-')[0]
        num_qubits = int(instance_info.split('-')[1])
        delta = float(instance_info.split('-')[2])

        return Model(plan, num_qubits, problem_name, delta, initializer)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='eigensolver_xxz1d-4-1.0_30_10',
            model_init='kaiming_normal',
            batchnorm_init='uniform' # Does not matter, we don't have module named 'BatchNorm2d'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='placeholder',
            batch_size=3
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=1e-1,
            #weight_decay=1e-4,
            training_steps='100ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
