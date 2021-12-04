
import torch
from torch import nn as nn

from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get


class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320)
        self.kernel_hidden = 1
        self.affine_eps = 0.0001
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)
        self.opt = opt
        self.le_curve = opt['le_curve'] if opt['le_curve'] is not None else False
        if self.le_curve:
            self.fCurve = self.F(in_channels=self.in_channels_rrdb,
                                 out_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 kernel_hidden=self.kernel_hidden,
                                 n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)

            # Curve conditional
            if self.le_curve:
                # logdet = logdet + thops.sum(torch.log(torch.sigmoid(z) * (1 - torch.sigmoid(z))), dim=[1, 2, 3])
                # z = torch.sigmoid(z)
                # alpha = self.fCurve(ft)
                # alpha = (torch.tanh(alpha + 2.) + self.affine_eps)
                # logdet = logdet + thops.sum(torch.log((1 + alpha - 2 * z * alpha).abs()), dim=[1, 2, 3])
                # z = z + alpha * z * (1 - z)

                alpha = self.fCurve(ft)
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                alpha = torch.relu(alpha) + self.affine_eps
                logdet = logdet + thops.sum(torch.log(alpha * torch.pow(z.abs(), alpha - 1)) + self.affine_eps)
                z = torch.pow(z.abs(), alpha) * z.sign()

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift
            z2 = z2 * scale

            logdet = logdet + self.get_logdet(scale)
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 / scale
            z2 = z2 - shift
            z = thops.cat_feature(z1, z2)
            logdet = logdet - self.get_logdet(scale)

            # Curve conditional
            if self.le_curve:
                # alpha = self.fCurve(ft)
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                # z = (1 + alpha) / alpha - (
                #             alpha + torch.pow(2 * alpha - 4 * alpha * z + torch.pow(alpha, 2) + 1, 0.5) + 1) / (
                #             2 * alpha)
                # z = torch.log((z / (1 - z)).clamp(1 / 1000, 1000))

                alpha = self.fCurve(ft)
                alpha = torch.relu(alpha) + self.affine_eps
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                z = torch.pow(z.abs(), 1 / alpha) * z.sign()

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)
