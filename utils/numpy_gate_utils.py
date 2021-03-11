### numpy implementation
import numpy as np
from numpy import kron
import functools

pauli_matrices = np.array((
    ((0, 1), (1, 0)),
    ((0, -1j), (1j, 0)),
    ((1, 0), (0, -1))
    ))
(_x, _y, _z) = pauli_matrices
_h = np.array([[1,1],[1,-1]]) / np.sqrt(2)
_phase = np.array([[1,0],[0,1j]])

_xx = np.kron(_x, _x)
_yy = np.kron(_y, _y)
_zz = np.kron(_z,_z)
_hh = np.kron(_h,_h)

_ix = np.kron(np.eye(2),_x)
_xi = np.kron(_x,np.eye(2))
_iy = np.kron(np.eye(2),_y)
_yi = np.kron(_y,np.eye(2))
_iz = np.kron(np.eye(2),_z)
_zi = np.kron(_z,np.eye(2))
_ih = np.kron(np.eye(2),_h)
_hi = np.kron(_h,np.eye(2))

_cz = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,-1]
    ])

_cnot = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
    ])

#### Helper functions for operators
def tensor(op_list):
    return functools.reduce(kron, op_list, 1)


# \sum_i X_i
def get_sumX(num_qubits):
    dim = 2 ** num_qubits
    all_dims = [2 ** l for l in range(num_qubits)]

    op = np.zeros((dim, dim))

    for i in range(num_qubits):
        op = op + tensor([np.eye(all_dims[i]), _x, np.eye(all_dims[num_qubits - i - 1])])
    return op

# \sum Z_{2i}Z_{2i+1} and \sum Z_{2i+1}Z_{2i+2}
def get_sumZZ(num_qubits, offset=0, ring=True):
    assert num_qubits % 2 == 0
    dim = 2 ** num_qubits
    all_dims = [2 ** l for l in range(num_qubits)]

    op = np.zeros((dim, dim))

    for i in np.arange(0, num_qubits, 2):
        j0 = (i + offset) % num_qubits
        j1 = (i + offset + 1) % num_qubits
        if not ring and j1 == 0:
            break
        op_list = [np.eye(2) for _ in range(num_qubits)]
        op_list[j0] = _z
        op_list[j1] = _z
        op = op + tensor(op_list)
    return op

def get_all0_initial_state(num_qubits):
    init_state = np.zeros((2 ** num_qubits, 1))
    init_state[0] = 1.0
    return init_state
    #return tensor([np.array([[1],[0]])] * num_qubits)

### MORE GENERAL FUNCS
def get_single_qubit_gate_i(single_gate, i, num_qubits):
    dim = 2 ** num_qubits
    all_dims = [2 ** l for l in range(num_qubits)]

    op = tensor([np.eye(all_dims[i]), single_gate, np.eye(all_dims[num_qubits - i - 1])])
    return op

def get_two_qubit_gate_ring(two_qubit_gate, num_qubits):
    dim = 2 ** num_qubits
    all_dims = [2 ** l for l in range(num_qubits)]

    op = np.eye(dim)
    for i in range(num_qubits-1):
        op_list = [np.eye(2)] * i + [two_qubit_gate] + [np.eye(2)] * (num_qubits - i - 2)
        op = tensor(op_list) @ op
    last_op = get_last_two_qubit_gate(two_qubit_gate, num_qubits)
    op = last_op @ op
    return op


def get_last_two_qubit_gate(two_qubit_gate, num_qubits):
    dim = 2 ** num_qubits
    all_dims = [2 ** l for l in range(num_qubits)]
    op = np.zeros((dim,dim))
    for a in range(4):
        for b in range(4):
            element = two_qubit_gate[a,b]
            a1, a2 = a // 2, a % 2
            b1, b2 = b // 2, b % 2
            m1 = np.zeros((2,2))
            m1[a1,b1] = 1.0
            m2 = np.zeros((2,2))
            m2[a2,b2] = 1.0
            op += element * tensor([m2] + [np.eye(2)] * (num_qubits - 2) + [m1])
    return op

def get_measurement_controlling_additional_qubit(two_qubit_gate, num_qubits, measurement, init_state=None):
    '''
    init_state: initial state of the single controlled qubit
    '''
    dim = 2 ** num_qubits
    all_dims = [2 ** l for l in range(num_qubits)]

    # initializing the unitary on (num_qubit + 1) qubits
    U = np.eye(2 * dim)
    # applying the two-qubit gate on the i-th and the last qubits
    for i in range(num_qubits):
        op = np.zeros((2 * dim, 2 * dim))
        for a in range(4):
            for b in range(4):
                element = two_qubit_gate[a,b]
                a1, a2 = a // 2, a % 2
                b1, b2 = b // 2, b % 2
                m1 = np.zeros((2,2))
                m1[a1,b1] = 1.0
                m2 = np.zeros((2,2))
                m2[a2,b2] = 1.0
                op += element * tensor([np.eye(2)] * i + [m1] + [np.eye(2)] * (num_qubits - 1 - i) + [m2])
        U = op @ U

    M = U.conj().T @ tensor([np.eye(dim), measurement]) @ U

    # partial trace
    if init_state is None:
        init_state = np.array([[1.0], [0.0]])
    V = tensor([np.eye(dim), init_state])
    return V.conj().T @ M @ V

'''
Calculating a list of operators, ordered from measurement to density matrices
'''
# Hardware efficient ansatz
def get_single_layer(gate, num_parameters):
    '''
    example: get_single_layer(tensor([_x, _y]), 3)
	    ->
	    |----X\otimes Y----|
	    |----X\otimes Y----|
	    |----X\otimes Y----|
    '''
    dim = gate.shape[0]
    op_list = []
    for l in range(num_parameters):
        op_list.append(
            tensor([np.eye(dim ** l), gate, np.eye(dim ** (num_parameters - 1 - l))])
        )
    return op_list

def get_list_of_ops_HEVA(single_layer_op_list, entangling_gate, num_layers):
    '''
    example:
	    single_layer_op_list = get_single_layer(_y, 3) + get_single_layer(_x, 3)
	    get_list_of_ops_HEVA(single_layer_op_list, U, 3)
	    ->
	    |--X--Y--|   |--X--Y--|   |--X--Y--|
	    |--X--Y--| U |--X--Y--| U |--X--Y--|
	    |--X--Y--|   |--X--Y--|   |--X--Y--|
    '''
    # repeat with entangling_gate
    num_gates_per_layer = len(single_layer_op_list)
    layer_list = single_layer_op_list

    op_list = []
    for l in range(num_layers):
        op_list = op_list + layer_list
        for k in range(num_gates_per_layer):
            layer_list[k] = entangling_gate @ layer_list[k] @ entangling_gate.conj().T
    return op_list
