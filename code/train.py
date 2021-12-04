import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import sys
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import options.options as option
from utils import util
from data import create_dataloader
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths
from data.LoL_dataset import LoL_Dataset, LoL_Dataset_v2
from torchvision.utils import save_image
import torchvision.transforms as T

to_tensor = T.ToTensor()
to_cv2_image = lambda x: np.array(T.ToPILImage()(torch.clip(x, 0, 1)))


def getEnv(name): import os; return True if name in os.environ.keys() else False


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_deviceDistIterSampler(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def color_adjust(low_light, output, kernel_size=7):
    # low_light, output = to_tensor(low_light), to_tensor(output)
    mean_kernal = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    low_light_mean = mean_kernal(low_light)
    output_mean = mean_kernal(output)
    color_align_output = output * (low_light_mean / output_mean)
    return color_align_output  # to_cv2_image(color_align_output)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YMAL file.',
                            default='./confs/low-light-server-modified_encoder.yml' if sys.platform != 'win32' else './confs/LOL_smallNet.yml') #  './confs/LOLv2-pc_rebuttal.yml') # 
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tfboard', action='store_true')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        resume_state_path, _ = get_resume_paths(opt)

        # distributed resuming: all load into default GPU
        if resume_state_path is None:
            resume_state = None
        else:
            device_id = torch.cuda.current_device()
            resume_state = torch.load(resume_state_path,
                                      map_location=lambda storage, loc: storage.cuda(device_id))
            option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                # from torch.utils.tensorboard import SummaryWriter
                if sys.platform != 'win32':
                    from tensorboardX import SummaryWriter
                else:
                    from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboard import SummaryWriter
            conf_name = basename(args.opt).replace(".yml", "")
            exp_dir = opt['path']['experiments_root']
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    if opt['dataset'] == 'LoL':
        dataset_cls = LoL_Dataset
    elif opt['dataset'] == 'LoL_v2':
        dataset_cls = LoL_Dataset_v2
    else:
        raise NotImplementedError()

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = dataset_cls(opt=dataset_opt, train=True, all_opt=opt)
            train_loader = create_dataloader(True, train_set, dataset_opt, opt, None)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
        elif phase == 'val':
            val_set = dataset_cls(opt=dataset_opt, train=False, all_opt=opt)
            val_loader = create_dataloader(False, val_set, dataset_opt, opt, None)
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    #### create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)
    print("Parameters of full network %.4f and encoder %.4f"%(sum([m.numel() for m in model.netG.parameters()])/1e6, sum([m.numel() for m in model.netG.RRDB.parameters()])/1e6))
    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    timer = Timer()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()
    avg_psnr = best_psnr = -1
    for epoch in range(start_epoch, total_epochs + 1):
        timerData.tick()
        for _, train_data in enumerate(train_loader):
            timerData.tock()
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            if current_step<2:
                nll = 0
            else:
                nll = model.optimize_parameters(current_step)

            #### log
            def eta(t_iter):
                return (t_iter * (opt['train']['niter'] - current_step)) / 3600

            if current_step % opt['logger']['print_freq'] == 0 \
                    or current_step - (resume_state['iter'] if resume_state else 0) < 25:
                avg_time = timer.get_average_and_reset()
                avg_data_time = timerData.get_average_and_reset()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, t:{:.2e}, td:{:.2e}, eta:{:.2e}, nll:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate(), avg_time, avg_data_time,
                    eta(avg_time), nll)
                print(message)
            timer.tick()
            # Reduce number of logs
            if current_step % 5 == 0 and args.tfboard:
                tb_logger_train.add_scalar('loss/nll', nll, current_step)
                tb_logger_train.add_scalar('lr/base', model.get_current_learning_rate(), current_step)
                tb_logger_train.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                tb_logger_train.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                tb_logger_train.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)
                for k, v in model.get_current_log().items():
                    tb_logger_train.add_scalar(k, v, current_step)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0
                nlls = []
                line = ''
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], str(current_step))
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    nll = model.test()
                    if nll is None:
                        nll = 0
                    nlls.append(nll)

                    visuals = model.get_current_visuals()

                    normal_img = None
                    # Save noramlly-exposed images for reference
                    if hasattr(model, 'heats') and model.heats is not None:
                        for heat in model.heats:
                            for i in range(model.n_sample):
                                normal_img = util.tensor2img(visuals['NORMAL', heat, i])  # uint8
                                save_img_path = os.path.join(img_dir,
                                                             '{:s}_{:09d}_h{:03d}_s{:d}.png'.format(img_name,
                                                                                                    current_step,
                                                                                                    int(heat * 100), i))
                                util.save_img(normal_img, save_img_path)
                    else:
                        if opt['align_color_from_lr']:
                            visuals['NORMAL'] = color_adjust(visuals['LQ'], visuals['NORMAL'], 11)
                        if opt['encode_color_map']:
                            color_lr, color_gt = model.get_color_map()
                            save_image(torch.cat([color_lr, color_gt], dim=0), os.path.join(img_dir,
                                                                                            'colormap_{:s}.png'.format(
                                                                                                img_name)), normalize=True)
                        normal_img = util.tensor2img(visuals['NORMAL'])  # uint8
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        # util.save_img(sr_img, save_img_path)
                    assert normal_img is not None

                    # Save LQ images for reference
                    save_img_path_lq = os.path.join(opt['path']['val_images'], 'low_light',
                                                    '{:s}_LQ.png'.format(img_name))
                    if not os.path.isfile(save_img_path_lq):
                        lq_img = util.tensor2img(visuals['LQ'])  # uint8
                        util.save_img(
                            cv2.resize(lq_img, dsize=None, fx=opt['scale'], fy=opt['scale'],
                                       interpolation=cv2.INTER_NEAREST),
                            save_img_path_lq)

                    # Save GT images for reference
                    gt_img = util.tensor2img(visuals['GT'])  # uint8
                    save_img_path_gt = os.path.join(opt['path']['val_images'], 'ground_truth',
                                                    '{:s}_GT.png'.format(img_name))
                    if not os.path.isfile(save_img_path_gt):
                        util.save_img(gt_img, save_img_path_gt)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    normal_img = normal_img / 255.

                    cropped_sr_img = normal_img  # [crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img  # [crop_size:-crop_size, crop_size:-crop_size, :]

                    # We follow a similar way of 'Kind' to finetune the overall brightness as illustrated in Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py).
                    # A normally-exposed image can also be obtained without finetuning the global brightness and we can achvieve compatible performance in terms of SSIM and LPIPS.
                    mean_gray_out = cv2.cvtColor(normal_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                    mean_gray_gt = cv2.cvtColor(gt_img.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                    cropped_sr_img_adjust = np.clip(cropped_sr_img * (mean_gray_gt / mean_gray_out), 0, 1)
                    psnr = util.calculate_psnr(cropped_sr_img_adjust * 255, cropped_gt_img * 255)
                    util.save_img((cropped_sr_img_adjust * 255).astype(np.uint8), save_img_path)
                    avg_psnr += psnr
                    ssim = util.ssim(visuals['GT'].unsqueeze(0), visuals['NORMAL'].unsqueeze(0)).item()
                    avg_ssim += ssim
                    line += '%s %.5f %.5f\n' % (img_name, psnr, ssim)

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_nll = sum(nlls) / len(nlls)
                with open(os.path.join(opt['path']['val_images'], str(current_step), 'metrics.txt'), 'w') as f:
                    f.write(line)

                # log
                logger.info('# Validation # PSNR: {:.4e} SSIM: {:.4e}'.format(avg_psnr, avg_ssim))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} SSIM: {:.4e}'.format(
                    epoch, current_step, avg_psnr, avg_ssim))

                # tensorboard logger
                if args.tfboard:
                    tb_logger_valid.add_scalar('loss/psnr', avg_psnr, current_step)
                    tb_logger_valid.add_scalar('loss/ssim', avg_ssim, current_step)
                    tb_logger_valid.add_scalar('loss/nll', avg_nll, current_step)
                    tb_logger_train.flush()
                    tb_logger_valid.flush()

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
            ### save best model
            if avg_psnr > best_psnr:
                logger.info('Saving best models')
                model.save('best_psnr')
                best_psnr = avg_psnr
                # model.save_training_state(epoch, current_step)
            timerData.tick()

    with open(os.path.join(opt['path']['root'], "TRAIN_DONE"), 'w') as f:
        f.write("TRAIN_DONE")

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
