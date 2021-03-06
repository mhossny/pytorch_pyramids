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


class GaussianSmoothing(nn.Module):
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
        return self.conv(x, weight=self.weight, groups=self.groups),


class LPHP2DFilter(nn.Module):
    def __init__(self, imgsz=(1, 256, 256),
                 lpkrnlsz=3, lpkrnlsgm=1.):
        nn.Module.__init__(self)
        self.lpkrnlsz = lpkrnlsz
        self.lpkrnlsgm = lpkrnlsgm
        self.imgsz = imgsz

        self.filter = GaussianSmoothing(channels=imgsz[0],
                                        kernel_size=lpkrnlsz,
                                        sigma=lpkrnlsgm,
                                        dim=2, padding_mode='reflect')

    def forward(self, x, *args, **kwargs):
        lpx, *more = self.filter(x)
        hpx = x - lpx

        return lpx, hpx

        
class MSDPyramidNet(nn.Module):
    def __init__(self, imgsz=(1, 256, 256),
                 lpkrnlsz=3, lpkrnlsgm=1.,
                 pyrmdpth=5):
        nn.Module.__init__(self)
        self.imgsz = imgsz
        self.lpkrnlsz = lpkrnlsz
        self.lpkrnlsgm = lpkrnlsgm
        self.pyrmdpth = pyrmdpth

        self.pyramid = nn.ModuleList([LPHP2DFilter(imgsz, lpkrnlsz, lpkrnlsgm)
                                      for i in range(pyrmdpth)])
        
        self.sampleimg = torch.randn(self.imgsz).unsqueeze(0)
        self.samplefeatures = self.forward(self.sampleimg)[0]
        self.featuresize = self.samplefeatures.size()
        self.nfeatures = self.samplefeatures.numel()

    def test(self, visualise=False, *args, **kwargs):
        '''
        1. analyse a sample image via forward method
        2. reconstruct sample image
        3. compare x and \hat x
        '''
        xlps, xhps = self.forward(self.sampleimg)
        if visualise:
            if xlps.size(1) < 4:
                imcat(xlps)
                imcat(xhps)
            
                for xlp, xhp in zip(xlps.transpose(0, 1), xhps.transpose(0, 1)):
                    # lp/hp are packed into chnls
                    imcat(xlp)
                    imcat(xhp)

        err = torch.round((self.sampleimg - self.rebuild(xlps, xhps)).sum() * 10**5) / 10.**5
        print(err)
        assert err == 0, '(E) reconstrction from pyramid failed, err=%f' % err
        
    def rebuild(self, xlps, xhps, *args, **kwargs):
        # pdb.set_trace()
        newxhps = [xhps[0, i::self.imgsz[0]].sum(0).unsqueeze(0) for i in range(self.imgsz[0])]
        newxhps = torch.cat(newxhps, dim=0).unsqueeze(0)

        newxlps = xlps[0, -self.imgsz[0]:].unsqueeze(0)

        return newxhps + newxlps  # xhps.sum(dim=1) + xlps[0, -1]

    def transform(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)
    
    def itransform(self, xlps, xhps, *args, **kwargs):
        return self.rebuild(xlps, xhps, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        xlps, xhps = (), ()
        for i, module in enumerate(self.pyramid):
            xlp, xhp, *more = module(x, *args, **kwargs)
            xlps += xlp,
            xhps += xhp,
            x = xlp

        return torch.cat(xlps, dim=1), torch.cat(xhps, dim=1)


stock = {'GaussianSmoothing': GaussianSmoothing}
