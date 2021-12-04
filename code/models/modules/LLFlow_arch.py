


import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules.RRDBNet_arch import RRDBNet
from models.modules.ConditionEncoder import ConEncoder1, NoEncoder
from models.modules.FlowUpsamplerNet import FlowUpsamplerNet
import models.modules.thops as thops
import models.modules.flow as flow
from models.modules.color_encoder import ColorEncoder
from utils.util import opt_get
from models.modules.flow import unsqueeze2d, squeeze2d
from torch.cuda.amp import autocast

class LLFlow(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, scale=4, K=None, opt=None, step=None):
        super(LLFlow, self).__init__()
        self.crop_size = opt['datasets']['train']['GT_size']
        self.opt = opt
        self.quant = 255 if opt_get(opt, ['datasets', 'train', 'quant']) is \
                            None else opt_get(opt, ['datasets', 'train', 'quant'])
        if opt['cond_encoder'] == 'ConEncoder1':
            self.RRDB = ConEncoder1(in_nc, out_nc, nf, nb, gc, scale, opt)
        elif opt['cond_encoder'] ==  'NoEncoder':
            self.RRDB = None # NoEncoder(in_nc, out_nc, nf, nb, gc, scale, opt)
        elif opt['cond_encoder'] == 'RRDBNet':
            # if self.opt['encode_color_map']: print('Warning: ''encode_color_map'' is not implemented in RRDBNet')
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)
        else:
            print('WARNING: Cannot find the conditional encoder %s, select RRDBNet by default.' % opt['cond_encoder'])
            # if self.opt['encode_color_map']: print('Warning: ''encode_color_map'' is not implemented in RRDBNet')
            opt['cond_encoder'] = 'RRDBNet'
            self.RRDB = RRDBNet(in_nc, out_nc, nf, nb, gc, scale, opt)

        if self.opt['encode_color_map']:
            self.color_map_encoder = ColorEncoder(nf=nf, opt=opt)

        hidden_channels = opt_get(opt, ['network_G', 'flow', 'hidden_channels'])
        hidden_channels = hidden_channels or 64
        self.RRDB_training = True  # Default is true

        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        set_RRDB_to_train = False
        if set_RRDB_to_train and self.RRDB:
            self.set_rrdb_training(True)

        self.flowUpsamplerNet = \
            FlowUpsamplerNet((self.crop_size, self.crop_size, 3), hidden_channels, K,
                             flow_coupling=opt['network_G']['flow']['coupling'], opt=opt)
        self.i = 0
        if self.opt['to_yuv']:
            self.A_rgb2yuv = torch.nn.Parameter(torch.tensor([[0.299, -0.14714119, 0.61497538],
                                                              [0.587, -0.28886916, -0.51496512],
                                                              [0.114, 0.43601035, -0.10001026]]), requires_grad=False)
            self.A_yuv2rgb = torch.nn.Parameter(torch.tensor([[1., 1., 1.],
                                                              [0., -0.39465, 2.03211],
                                                              [1.13983, -0.58060, 0]]), requires_grad=False)
        if self.opt['align_maxpool']:
            self.max_pool = torch.nn.MaxPool2d(3)

    def set_rrdb_training(self, trainable):
        if self.RRDB_training != trainable:
            for p in self.RRDB.parameters():
                p.requires_grad = trainable
            self.RRDB_training = trainable
            return True
        return False

    def rgb2yuv(self, rgb):
        rgb_ = rgb.transpose(1, 3)  # input is 3*n*n   default
        yuv = torch.tensordot(rgb_, self.A_rgb2yuv, 1).transpose(1, 3)
        return yuv

    def yuv2rgb(self, yuv):
        yuv_ = yuv.transpose(1, 3)  # input is 3*n*n   default
        rgb = torch.tensordot(yuv_, self.A_yuv2rgb, 1).transpose(1, 3)
        return rgb

    @autocast()
    def forward(self, gt=None, lr=None, z=None, eps_std=None, reverse=False, epses=None, reverse_with_grad=False,
                lr_enc=None,
                add_gt_noise=False, step=None, y_label=None, align_condition_feature=False, get_color_map=False):
        if get_color_map:
            color_lr = self.color_map_encoder(lr)
            color_gt = nn.functional.avg_pool2d(gt, 11, 1, 5)
            color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
            return color_lr, color_gt
        if not reverse:
            if epses is not None and gt.device.index is not None:
                epses = epses[gt.device.index]
            return self.normal_flow(gt, lr, epses=epses, lr_enc=lr_enc, add_gt_noise=add_gt_noise, step=step,
                                    y_onehot=y_label, align_condition_feature=align_condition_feature)
        else:
            # assert lr.shape[0] == 1
            assert lr.shape[1] == 3 or lr.shape[1] == 6
            # assert lr.shape[2] == 20
            # assert lr.shape[3] == 20
            # assert z.shape[0] == 1
            # assert z.shape[1] == 3 * 8 * 8
            # assert z.shape[2] == 20
            # assert z.shape[3] == 20
            if reverse_with_grad:
                return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                         add_gt_noise=add_gt_noise)
            else:
                with torch.no_grad():
                    return self.reverse_flow(lr, z, y_onehot=y_label, eps_std=eps_std, epses=epses, lr_enc=lr_enc,
                                             add_gt_noise=add_gt_noise)

    def normal_flow(self, gt, lr, y_onehot=None, epses=None, lr_enc=None, add_gt_noise=True, step=None,
                    align_condition_feature=False):
        if self.opt['to_yuv']:
            gt = self.rgb2yuv(gt)
        if lr_enc is None and self.RRDB:
            lr_enc = self.rrdbPreprocessing(lr)

        logdet = torch.zeros_like(gt[:, 0, 0, 0])
        pixels = thops.pixels(gt)

        z = gt

        if add_gt_noise:
            # Setup
            noiseQuant = opt_get(self.opt, ['network_G', 'flow', 'augmentation', 'noiseQuant'], True)
            if noiseQuant:
                z = z + ((torch.rand(z.shape, device=z.device) - 0.5) / self.quant)
            logdet = logdet + float(-np.log(self.quant) * pixels)

        # Encode
        epses, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, gt=z, logdet=logdet, reverse=False, epses=epses,
                                              y_onehot=y_onehot)

        objective = logdet.clone()

        # if isinstance(epses, (list, tuple)):
        #     z = epses[-1]
        # else:
        #     z = epses
        z = epses
        if 'avg_color_map' in self.opt.keys() and self.opt['avg_color_map']:
            if 'avg_pool_color_map' in self.opt.keys() and self.opt['avg_pool_color_map']:
                mean = squeeze2d(F.avg_pool2d(lr_enc['color_map'], 7, 1, 3), 8) if random.random() > self.opt[
                    'train_gt_ratio'] else squeeze2d(F.avg_pool2d(
                    gt / (gt.sum(dim=1, keepdims=True) + 1e-4), 7, 1, 3), 8)
        else:
            if self.RRDB is not None:
                mean = squeeze2d(lr_enc['color_map'], 8) if random.random() > self.opt['train_gt_ratio'] else squeeze2d(
                gt/(gt.sum(dim=1, keepdims=True) + 1e-4), 8)
            else:
                mean = squeeze2d(lr[:,:3],8)
        objective = objective + flow.GaussianDiag.logp(mean, torch.tensor(0.).to(z.device), z)

        nll = (-objective) / float(np.log(2.) * pixels)
        if self.opt['encode_color_map']:
            color_map = self.color_map_encoder(lr)
            color_gt = nn.functional.avg_pool2d(gt, 11, 1, 5)
            color_gt = color_gt / torch.sum(color_gt, 1, keepdim=True)
            color_loss = (color_gt - color_map).abs().mean()
            nll = nll + color_loss
        if align_condition_feature:
            with torch.no_grad():
                gt_enc = self.rrdbPreprocessing(gt)
            for k, v in gt_enc.items():
                if k in ['fea_up-1']:  # ['fea_up2','fea_up1','fea_up0','fea_up-1']:
                    if self.opt['align_maxpool']:
                        nll = nll + (self.max_pool(gt_enc[k]) - self.max_pool(lr_enc[k])).abs().mean() * (
                            self.opt['align_weight'] if self.opt['align_weight'] is not None else 1)
                    else:
                        nll = nll + (gt_enc[k] - lr_enc[k]).abs().mean() * (
                            self.opt['align_weight'] if self.opt['align_weight'] is not None else 1)
        if isinstance(epses, list):
            return epses, nll, logdet
        return z, nll, logdet

    def rrdbPreprocessing(self, lr):
        rrdbResults = self.RRDB(lr, get_steps=True)
        block_idxs = opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'blocks']) or []
        if len(block_idxs) > 0:
            low_level_features = [rrdbResults["block_{}".format(idx)] for idx in block_idxs]
            # low_level_features.append(rrdbResults['color_map'])
            concat = torch.cat(low_level_features, dim=1)

            if opt_get(self.opt, ['network_G', 'flow', 'stackRRDB', 'concat']) or False:
                keys = ['last_lr_fea', 'fea_up1', 'fea_up2', 'fea_up4']
                if 'fea_up0' in rrdbResults.keys():
                    keys.append('fea_up0')
                if 'fea_up-1' in rrdbResults.keys():
                    keys.append('fea_up-1')
                for k in keys:
                    h = rrdbResults[k].shape[2]
                    w = rrdbResults[k].shape[3]
                    rrdbResults[k] = torch.cat([rrdbResults[k], F.interpolate(concat, (h, w))], dim=1)
        return rrdbResults

    def get_score(self, disc_loss_sigma, z):
        score_real = 0.5 * (1 - 1 / (disc_loss_sigma ** 2)) * thops.sum(z ** 2, dim=[1, 2, 3]) - \
                     z.shape[1] * z.shape[2] * z.shape[3] * math.log(disc_loss_sigma)
        return -score_real

    def reverse_flow(self, lr, z, y_onehot, eps_std, epses=None, lr_enc=None, add_gt_noise=True):

        logdet = torch.zeros_like(lr[:, 0, 0, 0])
        pixels = thops.pixels(lr) * self.opt['scale'] ** 2

        if add_gt_noise:
            logdet = logdet - float(-np.log(self.quant) * pixels)

        if lr_enc is None and self.RRDB:
            lr_enc = self.rrdbPreprocessing(lr)
        if self.opt['cond_encoder'] == "NoEncoder":
            z = squeeze2d(lr[:,:3],8)
        else:
            if 'avg_color_map' in self.opt.keys() and self.opt['avg_color_map']:
                z = squeeze2d(F.avg_pool2d(lr_enc['color_map'], 7, 1, 3), 8)
            else:
                z = squeeze2d(lr_enc['color_map'], 8)
        x, logdet = self.flowUpsamplerNet(rrdbResults=lr_enc, z=z, eps_std=eps_std, reverse=True, epses=epses,
                                          logdet=logdet)
        if self.opt['encode_color_map']:
            color_map = self.color_map_encoder(lr)
            color_out = nn.functional.avg_pool2d(x, 11, 1, 5)
            color_out = color_out / torch.sum(color_out, 1, keepdim=True)
            x = x * (color_map / color_out)
        if self.opt['to_yuv']:
            x = self.yuv2rgb(x)
        return x, logdet
