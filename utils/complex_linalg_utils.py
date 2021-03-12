import numpy as np
import torch
from torch import autograd, nn


def cmm(input1,input2):
    return torch.view_as_complex(torch.stack(
        (
            torch.matmul(input1.real, input2.real) - torch.matmul(input1.imag, input2.imag),
            torch.matmul(input1.real, input2.imag) + torch.matmul(input1.imag, input2.real)
        ),
        dim=-1))
class ComplexExpJ(autograd.Function):
    #real thetas
    @staticmethod
    def forward(ctx, theta):
        output = torch.cos(theta) + 1j * torch.sin(theta)
        ctx.save_for_backward(theta, output)
        return output
    def backward(ctx, grad_out):
        theta, output = ctx.saved_tensors
        grad_theta = (grad_out.conj()) * 1j * output
        return grad_theta.real
cexpj = ComplexExpJ.apply

class ComplexMatExp(autograd.Function):
    #real thetas, hermitian ops
    # theta shape: (num_ops,)
    #torch.linalg.eigh
    @staticmethod
    def forward(ctx, thetas, ops):
        ops = [theta * op for theta, op in zip(thetas, ops)]
        output = torch.cos(theta) + 1j * torch.sin(theta)
        ctx.save_for_backward(theta, output)
        return output
    def backward(ctx, grad_out):
        theta, output = ctx.saved_tensors
        grad_theta = (grad_out.conj()) * 1j * output
        return grad_theta.real
cexpj = ComplexExpJ.apply

##  For nightly build pytorch
if torch.__version__ != '1.7.0' and torch.__version__ != '1.7.1+cu110':
    #print('Using nightly build..')
    cmm = torch.matmul
    cexpj = lambda x: torch.exp(1j * x)

def expjm(input, theta, decomp_op):
    output = cmm(decomp_op[1].conj().t(), input)
    output = cexpj(theta * decomp_op[0]).mul(output.transpose(-2,-1)).transpose(-2,-1)
    output = cmm(decomp_op[1], output)
    return output

def batch_expjm(input, theta, decomp_op):
    # theta shape: (M,)
    # d shape: (dim,)
    # v shape: (dim,dim)
    # input shape: (N, M, dim, dim_in=1) or (N, 1, dim, dim_in=1)
    output = cmm(decomp_op[1].conj().t(), input) # (N, M, dim, dim_in) or (N, 1, dim, dim_in=1)
    theta_d = cexpj(theta.unsqueeze(-1) * decomp_op[0].unsqueeze(0)).unsqueeze(0).unsqueeze(-1) # (1, M, dim, 1)
    output = theta_d * output # (N, M, dim, dim_in)
    output = cmm(decomp_op[1], output) # (N, M, dim, dim_in)
    return output

def expectation(input, observable=None):
    if observable is None:
        output = cmm(input.conj().transpose(-2,-1), input)
    else:
        output = cmm(input.conj().transpose(-2,-1), cmm(observable, input))
    return output.real.squeeze()

