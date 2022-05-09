import os
import torch.nn as nn
import torch.optim as optim
import torchvision
from data.noisy_cifar import NoisyCIFAR10, NoisyCIFAR100
from torch.optim.lr_scheduler import LambdaLR
import math


def build_cifar100n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio,logger = None):
    train_data = NoisyCIFAR100(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True,logger = logger)
    test_data = NoisyCIFAR100(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True,logger = logger)
    return train_data,test_data

def build_cifar10n_dataset(root, train_transform, test_transform, noise_type, openset_ratio, closeset_ratio,logger = None):
    train_data = NoisyCIFAR10(root, train=True, transform=train_transform, download=False, noise_type=noise_type, closeset_ratio=closeset_ratio,
                               openset_ratio=openset_ratio, verbose=True,logger = logger)
    test_data = NoisyCIFAR10(root, train=False, transform=test_transform, download=False, noise_type='clean', closeset_ratio=closeset_ratio,
                              openset_ratio=openset_ratio, verbose=True,logger = logger)
    return train_data,test_data

# optimizer, scheduler -------------------------------------------------------------------------------------------------------------------------------
def build_sgd_optimizer(params, lr, weight_decay):
    return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


def build_adam_optimizer(params, lr):
    return optim.Adam(params, lr=lr, betas=(0.9, 0.999))


def build_cosine_lr_scheduler(optimizer, total_epochs):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_opt_sched(model,lr,warmup,total_steps,weight_decay,opt = 'sgd'):
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if opt == 'sgd':
        optimizer = optim.SGD(grouped_parameters, lr=lr,
                              momentum=0.9, nesterov=True)
    elif opt == 'adam':
        optimizer = optim.Adam(grouped_parameters, lr=2e-3)

    # args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup, total_steps)

    return  optimizer, scheduler
