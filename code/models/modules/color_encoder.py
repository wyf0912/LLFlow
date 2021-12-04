


import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from utils.util import opt_get
from models.modules.base_layers import *


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ColorEncoder(nn.Module):
    def __init__(self, nf, opt=None):
        self.opt = opt
        super(ColorEncoder, self).__init__()
        self.conv_input = Conv2D(3, nf)
        # top path build Reflectance map
        self.maxpool_r1 = MaxPooling2D()
        self.conv_r1 = Conv2D(nf, nf * 2)
        self.maxpool_r2 = MaxPooling2D()
        self.conv_r2 = Conv2D(nf * 2, nf * 4)
        self.deconv_r1 = ConvTranspose2D(nf * 4, nf * 2)
        self.concat_r1 = Concat()
        self.conv_r3 = Conv2D(nf * 4, nf * 2)
        self.deconv_r2 = ConvTranspose2D(nf * 2, nf)
        self.concat_r2 = Concat()
        self.conv_r4 = Conv2D(nf * 2, nf)
        self.conv_r5 = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        # self.R_out = nn.Sigmoid()
        self.R_out = nn.Sigmoid()# (negative_slope=0.2, inplace=True)
        # bottom path build Illumination map
        # self.conv_i1 = Conv2D(nf, nf)
        # self.concat_i1 = Concat()
        # self.conv_i2 = nn.Conv2d(nf * 2, 1, kernel_size=3, padding=1)
        # self.I_out = nn.Sigmoid()

    def forward(self, x, get_steps=False):
        assert not get_steps

        # x = torch.cat([x, color_x], dim=1)
        conv_input = self.conv_input(x)
        # build Reflectance map
        maxpool_r1 = self.maxpool_r1(conv_input)
        conv_r1 = self.conv_r1(maxpool_r1)
        maxpool_r2 = self.maxpool_r2(conv_r1)
        conv_r2 = self.conv_r2(maxpool_r2)
        deconv_r1 = self.deconv_r1(conv_r2)
        concat_r1 = self.concat_r1(conv_r1, deconv_r1)
        conv_r3 = self.conv_r3(concat_r1)
        deconv_r2 = self.deconv_r2(conv_r3)
        concat_r2 = self.concat_r2(conv_input, deconv_r2)
        conv_r4 = self.conv_r4(concat_r2)
        conv_r5 = self.conv_r5(conv_r4)
        R_out = self.R_out(conv_r5)
        color_x = nn.functional.avg_pool2d(R_out, self.opt['avg_kernel_size'], 1, self.opt['avg_kernel_size']//2)
        # color_x = color_x / torch.sum(color_x, 1, keepdim=True)
        # build Illumination map
        # conv_i1 = self.conv_i1(conv_input)
        # concat_i1 = self.concat_i1(conv_r4, conv_i1)
        # conv_i2 = self.conv_i2(concat_i1)
        # I_out = self.I_out(conv_i2)

        return color_x
