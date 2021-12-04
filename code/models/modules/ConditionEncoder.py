

from torchvision.utils import save_image
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
from utils.util import opt_get
from models.modules.flow import Conv2dZeros


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
        # gamma = torch.sigmoid(self.conv5(torch.cat((x, x1, x2, x3, x4), 1)))
        # x = torch.sigmoid(x)
        # return x + gamma * x * (1 - x)


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


class ConEncoder1(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        self.gray_map_bool = False
        self.concat_color_map = False
        if opt['concat_histeq']:
            in_nc = in_nc + 3
        if opt['concat_color_map']:
            in_nc = in_nc + 3
            self.concat_color_map = True
        if opt['gray_map']:
            in_nc = in_nc + 1
            self.gray_map_bool = True
        in_nc = in_nc + 6
        super(ConEncoder1, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 3, 1, 1),nn.Sigmoid())

    def forward(self, x, get_steps=False):
        if self.gray_map_bool:
            x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
        if self.concat_color_map:
            x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)

        raw_low_input = x[:, 0:3].exp()
        # fea_for_awb = F.adaptive_avg_pool2d(fea_down8, 1).view(-1, 64)
        awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
        low_after_awb = raw_low_input * awb_weight
        # import pdb
        # pdb.set_trace()
        color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
        dx, dy = self.gradient(color_map)
        noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]
        # color_map = self.fine_tune_color_map(torch.cat([color_map, noise_map], dim=1))

        fea = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea = self.lrelu(fea)
        fea = self.conv_second(fea)
        fea_head = F.max_pool2d(fea, 2)

        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_results = {}
        fea = fea_head
        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        # fea = F.max_pool2d(fea, 2)
        fea_down2 = fea_head + trunk

        fea_down4 = self.downconv1(F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                                                 recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)

        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down8)

        # fea_down16 = self.downconv3(
        #     F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down16)

        results = {'fea_up0': fea_down8,
                   'fea_up1': fea_down4,
                   'fea_up2': fea_down2,
                   'fea_up4': fea_head,
                   'last_lr_fea': fea_down4,
                   'color_map': self.fine_tune_color_map(F.interpolate(fea_down2, scale_factor=2))
                   }

        # 'color_map': color_map}  # raw

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return None

    def gradient(self, x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)


class NoEncoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, opt=None):
        self.opt = opt
        self.gray_map_bool = False
        self.concat_color_map = False
        if opt['concat_histeq']:
            in_nc = in_nc + 3
        if opt['concat_color_map']:
            in_nc = in_nc + 3
            self.concat_color_map = True
        if opt['gray_map']:
            in_nc = in_nc + 1
            self.gray_map_bool = True
        in_nc = in_nc + 6
        super(NoEncoder, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.scale = scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### downsampling
        self.downconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.downconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(nn.Conv2d(nf, 3, 1, 1),nn.Sigmoid())

    def forward(self, x, get_steps=False):
        if self.gray_map_bool:
            x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
        if self.concat_color_map:
            x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)

        raw_low_input = x[:, 0:3].exp()
        # fea_for_awb = F.adaptive_avg_pool2d(fea_down8, 1).view(-1, 64)
        awb_weight = 1  # (1 + self.awb_para(fea_for_awb).unsqueeze(2).unsqueeze(3))
        low_after_awb = raw_low_input * awb_weight
        # import pdb
        # pdb.set_trace()
        color_map = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
        dx, dy = self.gradient(color_map)
        noise_map = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]
        # color_map = self.fine_tune_color_map(torch.cat([color_map, noise_map], dim=1))

        fea = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea = self.lrelu(fea)
        fea = self.conv_second(fea)
        fea_head = F.max_pool2d(fea, 2)

        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        block_results = {}
        fea = fea_head
        for idx, m in enumerate(self.RRDB_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
        trunk = self.trunk_conv(fea)
        # fea = F.max_pool2d(fea, 2)
        fea_down2 = fea_head + trunk

        fea_down4 = self.downconv1(F.interpolate(fea_down2, scale_factor=1 / 2, mode='bilinear', align_corners=False,
                                                 recompute_scale_factor=True))
        fea = self.lrelu(fea_down4)

        fea_down8 = self.downconv2(
            F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down8)

        # fea_down16 = self.downconv3(
        #     F.interpolate(fea, scale_factor=1 / 2, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        # fea = self.lrelu(fea_down16)

        results = {'fea_up0': fea_down8*0,
                   'fea_up1': fea_down4*0,
                   'fea_up2': fea_down2*0,
                   'fea_up4': fea_head*0,
                   'last_lr_fea': fea_down4*0,
                   'color_map': self.fine_tune_color_map(F.interpolate(fea_down2, scale_factor=2))*0
                   }

        # 'color_map': color_map}  # raw

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return None

    def gradient(self, x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)
