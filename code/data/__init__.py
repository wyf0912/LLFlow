


'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(train, dataset, dataset_opt, opt=None, sampler=None):
    gpu_ids = opt.get('gpu_ids', None)
    gpu_ids = gpu_ids if gpu_ids else []
    num_workers = dataset_opt['n_workers'] * (len(gpu_ids)+1)
    batch_size = dataset_opt['batch_size']
    shuffle = True
    if train:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           num_workers=num_workers, sampler=sampler, drop_last=False,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    print(dataset_opt)
    mode = dataset_opt['mode']
    if mode == 'LoL':
        from data.LoL_dataset import LoL_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
