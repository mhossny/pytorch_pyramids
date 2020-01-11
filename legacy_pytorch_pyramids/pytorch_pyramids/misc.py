#!/usr/bin/python3 -Wignore::DeprecationWarning

from torch.nn import init
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
import math
import copy
import numbers
from math import log2, floor

import numpy as np


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.orgsz = None
        pass

    def revert(self, x, *args, **kwargs):
        assert self.orgsz, '(E) x not flattenned'
        
        return x.view(self.orgsz)
    
    def forward(self, x, *args, **kwargs):
        self.orgsz = x.size()
        return x.view(x.size(0), -1), self.orgsz


class GaussianFilter(nn.Module):
    """
    Many thanks to Adrian Sahlman (tetratrio,
    https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8).

    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels=3, kernel_size=3, sigma=1., dim=2, padding_mode='reflect'):
        nn.Module.__init__(self)
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # pdb.set_trace()
        self.padding = kernel_size[0] // 2
        self.padding_mode = padding_mode
            
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (1 / (std * math.sqrt(2 * math.pi)) *
                       torch.exp(-((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

        for prm in self.parameters():
            prm.require_grad = False

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        pd = self.padding
        x = F.pad(input, (pd, pd, pd, pd), mode=self.padding_mode)
        # pdb.set_trace()
        return self.conv(x, weight=self.weight, groups=self.groups),


stock = {'Flatten': Flatten}
