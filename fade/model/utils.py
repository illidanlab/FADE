"""
source: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
"""

# from PIL import Image

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def loop_iterable(iterable):
    while True:
        yield from iterable


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)

    Refer to: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def freeze_model(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze

