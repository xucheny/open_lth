import numpy as np
import torch
from torch import nn

_CTYPE=torch.cfloat
_RTYPE=torch.float
from utils.complex_linalg_utils import cmm, cexpj, expjm, batch_expjm, expectation
from utils.numpy_gate_utils import tensor, _x, _z

class Energy(nn.Module):
    def __init__(self, observable):
        super(Energy, self).__init__()
        self.register_buffer('ob', torch.tensor(observable, dtype=_CTYPE))
        vals, vecs = np.linalg.eigh(observable)
        self.optimal_value = vals[0]

    def forward(self, state):
        return expectation(state, self.ob) - self.optimal_value

class PlaceholderLoss(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(PlaceholderLoss, self).__init__(size_average, reduce, reduction)
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return torch.mean(input)
        elif self.reduction == 'sum':
            return torch.sum(input)
        else:
            raise ValueError("reduction for PlaceholderLoss should be either 'mean' or 'sum'.")

class HamiltonianVariationalAnsatz(nn.Module):
    '''
    Operations allow batching of M sets of parameters and N different input states:
    params: (M, num_layers, num_ops)
    input: (N, 1, dim, 1)
    output: (N, M, dim, 1)
    Here we restrict M = N = 1
    '''

    def __init__(self, parameterized_ops, num_layers, input_state):
        super(HamiltonianVariationalAnsatz, self).__init__()
        self.num_layers = num_layers
        self.num_ops = len(parameterized_ops)

        # register parameterized ops
        for op_ind, op in enumerate(parameterized_ops):
            d, v = np.linalg.eigh(op)
            self.register_buffer('op_d'+str(op_ind), torch.tensor(d, dtype=_RTYPE))
            self.register_buffer('op_v'+str(op_ind), torch.tensor(v, dtype=_CTYPE))
        # register input state
        self.register_buffer('input', torch.tensor(input_state, dtype=_CTYPE))

        # register parameters
        self.params = nn.Parameter(torch.Tensor(1, self.num_layers, self.num_ops))


    def forward(self):
        state = self.input
        for layer_ind in range(self.num_layers):
            for op_ind in range(self.num_ops):
                state = batch_expjm(state,
                        -self.params[:,layer_ind, op_ind],
                        [getattr(self, 'op_d'+str(op_ind)), getattr(self, 'op_v'+str(op_ind))]
                        )
        return state

def get_TFIM_setup(num_qubits, g):
    dim = 2 ** num_qubits
    ring_graph = [(i, i+1) for i in range(num_qubits - 1)]
    Hzz = np.zeros((dim, dim), dtype=np.complex128)
    for edge in ring_graph:
        add_gate = [np.eye(2)] * num_qubits
        add_gate[edge[0]] = _z
        add_gate[edge[1]] = _z
        Hzz = Hzz + tensor(add_gate)

    Hx = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(num_qubits):
        add_gate = [np.eye(2)] * num_qubits
        add_gate[i] = _x
        Hx = Hx + tensor(add_gate)

    parameterized_ops = [Hx, Hzz]
    observable = Hzz + g * Hx
    input_state = tensor([np.array([1.0,0])] * num_qubits)
    input_state = input_state.reshape(1,1,dim,1)

    return parameterized_ops, observable, input_state


