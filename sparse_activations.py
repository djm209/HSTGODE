# -*- coding: utf-8 -*-
import torch
from torch.autograd import Function
import torch.nn as nn


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold 计算阈值
    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax
    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    #对输入的相似度进行降序排序，在dim的维度上,返回值；
    input_cumsum = input_srt.cumsum(dim) - 1
    #返回输入沿指定维度的累积和
    rhos = _make_ix_like(input, dim)
    #返回一个【1,1,1，dim】的序列tensor:1,2,3,...,input.size(dim),也就是公式中的K
    support = rhos * input_srt > input_cumsum#生成满足下式的mask
    #j*u_i-sum(u_i)>0,i=1...j
    support_size = support.sum(dim=dim).unsqueeze(dim)
    #j*u_i, j<k,求和
    
    #对j<k（按照mask）求和对参照论文中的算法
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    
    
    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, input, dim=0):
        """sparsemax: normalizing sparse transform (a la softmax)
        Parameters:
            input (Tensor): any shape
            dim: dimension along which to apply sparsemax
        Returns:
            output (Tensor): same shape as input
        """
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input -= max_val  # same numerical stability trick as for softmax
        tau, supp_size = _threshold_and_support(input, dim=dim)
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class LogSparsemax(nn.Module):

    def __init__(self, dim=0):
        self.dim = dim
        super(LogSparsemax, self).__init__()

    def forward(self, input):
        return torch.log(sparsemax(input, self.dim))