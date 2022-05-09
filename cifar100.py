import os
import pathlib
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchsummary
import copy
# from apex import amp
from utils.core import accuracy, evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger, print_to_logfile, print_to_console, step_flagging
from utils.ema import EMA
from utils.model import Model
from data.transform import TransformWSW
from PIL import ImageFile
from utils.utils import variable_to_numpy
ImageFile.LOAD_TRUNCATED_IMAGES = True

def adjust_lr_beta1(optimizer, lr, beta1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, 0.999)  # Only change beta1

def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    p = p + 1e-8
    q = q + 1e-8
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def main(cfg, device):
    init_seeds()
    cfg.use_fp16 = False if device.type == 'cpu' else cfg.use_fp16

    # logging ----------------------------------------------------------------------------------------------------------------------------------------
    logger_root = f'Results/{cfg.dataset}'
    if not os.path.isdir(logger_root):
        os.makedirs(logger_root, exist_ok=True)
    logtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if cfg.resume is None:
        result_dir = os.path.join(logger_root, f'{cfg.log}-{logtime}')
        logger = Logger(logging_dir=result_dir, DEBUG=False)
        logger.set_logfile(logfile_name='log.txt')
    else:
        result_dir = cfg.result_dir
        logger = Logger(logging_dir=cfg.result_dir, DEBUG=False)
        logger.set_logfile('log.txt')
    save_config(cfg, f'{result_dir}/config.cfg')
    save_params(cfg, f'{result_dir}/params.json', json_format=True)
    logger.debug(f'Result Path: {result_dir}')

    # model, optimizer, scheduler --------------------------------------------------------------------------------------------------------------------
    n_classes = int(cfg.n_classes * (1 - cfg.openset_ratio))
    
    print_to_console(f'> number of classes: {n_classes}', color='red')
    
    net = Model(arch=cfg.net, num_classes=n_classes, pretrained=cfg.pretrained).to(device)
    net_ema = Model(arch=cfg.net, num_classes=n_classes, pretrained=cfg.pretrained)

    # log network
    with open(f'{result_dir}/network.txt', 'w') as f:
        f.writelines(net.__repr__())

    # Adjust learning rate and betas for Adam Optimizer
    epoch_decay_start = 80
    mom1 = 0.9
    mom2 = 0.1
    lr_plan = [cfg.lr] * cfg.epochs
    beta1_plan = [mom1] * cfg.epochs
    for i in range(epoch_decay_start, cfg.epochs):
        lr_plan[i] = float(cfg.epochs - i) / (cfg.epochs - epoch_decay_start) * cfg.lr
        beta1_plan[i] = mom2

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    cifar_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # dataset, dataloader ----------------------------------------------------------------------------------------------------------------------------
    train_data,test_data = build_cifar100n_dataset(
        os.path.join(cfg.database, cfg.dataset), 
        TransformWSW(cifar100_mean,cifar100_std), 
        cifar_test_transform,
        noise_type=cfg.noise_type,
        openset_ratio=cfg.openset_ratio,
        closeset_ratio=cfg.closeset_ratio,
        logger = logger
    )
    train_data.control(mode='101')
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

    # if cfg.cosineAnnealing:
    # warmup_steps = cfg.warmup_epochs * len(train_loader)
    warmup_steps = 0
    total_steps = warmup_steps + (cfg.epochs - cfg.warmup_epochs) * cfg.steps
    print_to_console(f'total steps -- > {total_steps}',color='red')
    optimizer, scheduler = get_opt_sched(net,cfg.lr,warmup_steps,total_steps,cfg.weight_decay,opt='sgd')
    
    # opt_lvl = 'O1' if cfg.use_fp16 else 'O0'
    # [net, net_ema], optimizer = \
    #     amp.initialize([net.to(device), net_ema.to(device)], optimizer, \
    #         opt_level=opt_lvl,keep_batchnorm_fp32=None, loss_scale=None, verbosity=0)

    
    # meters -----------------------------------------------------------------------------------------------------------------------------------------
    train_loss = AverageMeter()
    epoch_train_time = AverageMeter()

    
    # resume -----------------------------------------------------------------------------------------------------------------------------------------
    if cfg.resume is not None:
        assert os.path.isfile(cfg.resume), 'no checkpoint.pth exists!'
        logger.debug(f'---> loading {cfg.resume} <---')
        checkpoint = torch.load(f'{cfg.resume}')
        start_epoch = checkpoint['epoch']
        # original net
        best_accuracy = checkpoint['best_accuracy']
        best_epoch = checkpoint['best_epoch']
        net.load_state_dict(checkpoint['state_dict'])
        # ema net
        # best_epoch_ema = checkpoint['best_epoch_ema']
        # best_accuracy_ema = checkpoint['best_accuracy_ema']
        # net_ema.load_state_dict(checkpoint['state_dict'])
        # optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_accuracy = 0.0
        best_accuracy_ema = 0.0
        best_epoch_ema = None
        best_epoch = None

    ema = EMA(net, alpha=0.99)
    ema.apply_shadow(net_ema)

    flag = 0
    cleanset,unlabeledset,unlabeledset_all = None,None,None
    # training ---------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        start_time = time.time()

        # pre-step in this epoch
        net.train()
        
        train_loss.reset()
        curr_lr = [group['lr'] for group in optimizer.param_groups]
        logger.debug(f'Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  Lr:[{curr_lr[0]:.5f}]')

        if epoch < cfg.warmup_epochs:
            
            
            for it, sample in enumerate(train_loader):
                s = time.time()
                
                optimizer.zero_grad()
                
                (input_w_plus, input_w), targets = sample
                input_w_plus,input_w = input_w_plus.to(device),input_w.to(device)
                targets = targets.to(device)
                N = input_w.size(0)
                
                input_all = torch.cat([input_w,input_w_plus],dim=0)
                logits_all = net(input_all,has_open=False)

                logits_w,logits_w_plus = logits_all[:N],logits_all[N:]
                
                probs_w = F.softmax(logits_w, dim=1)
                probs_w_plus = F.softmax(logits_w_plus, dim=1)
                
                # preds_close = probs_w.max(dim=1)[1]

                C = logits_w.shape[1]
                given_labels = torch.full(size=(N, C), fill_value=cfg.eps/(C - 1)).to(device)
                given_labels.scatter_(dim=1, index=torch.unsqueeze((targets), dim=1), value=1-cfg.eps)

                if flag == 0:
                    step_flagging(f'start the warm-up step for {cfg.warmup_epochs} epochs.')
                    flag += 1
                lx = F.cross_entropy(logits_all,targets.repeat(2),reduction='mean')
                # l_con = torch.mean(torch.pow(probs_w - probs_w_plus,2).sum(dim=1))
                # loss = lx + cfg.lamb_con * l_con
                loss = lx
                
                train_loss.update(loss.item(),N)
                
                # if cfg.use_fp16:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()
                optimizer.step()
                
                scheduler.step()

                # ema.update_params(net)
                # ema.apply_shadow(net_ema)    
                epoch_train_time.update(time.time() - s, 1)
                
                if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == len(train_loader)):
                    console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                    f"Iter:[{it + 1:>4d}/{len(train_loader):>4d}]  " \
                                    f"Loss:[{train_loss.avg:4.4f}]  " \
                                    f"{epoch_train_time.avg:6.2f} sec/iter"
                    logger.debug(console_content)
            
            # 在warmup epoch最后筛选clean样本，将其余样本都看做unlabeled样本
            if epoch == cfg.warmup_epochs - 1:
                # 不改变排列的顺序，即用squentialSample进行采样
                squential_loader = DataLoader(
                        train_data,
                        batch_size=cfg.batch_size,
                        num_workers=8,
                        drop_last=False,
                        shuffle=False)
                for batch_id, sample in enumerate(squential_loader):
                    (input_w_plus, input_w),targets = sample
                    input_w_plus,input_w = input_w_plus.to(device),input_w.to(device)
                    targets = targets.to(device)
                    
                    input_all = torch.cat([input_w,input_w_plus],dim=0)
                    logits_all = net(input_all,has_open=False)
                    logits_w_plus,logits_w = logits_all.chunk(2)
                    probability = (F.softmax(logits_w_plus, dim=1) + F.softmax(logits_w, dim=1)) / 2
                    label_smooth = torch.full(size=(input_w.size(0), C), fill_value=cfg.eps/(C - 1)).to(device)
                    label_smooth.scatter_(dim=1, index=torch.unsqueeze((targets), dim=1), value=1-cfg.eps)
                    prob_clean = 1 - js_div(probability, label_smooth)
                    clean_idx = prob_clean > cfg.tau_clean
                    if batch_id == 0:
                        clean_all = clean_idx
                    else:
                        clean_all = torch.cat([clean_all,clean_idx],0)
                    
                clean_all = variable_to_numpy(clean_all)
                cleanset = copy.deepcopy(train_data)
                unlabeledset = copy.deepcopy(train_data)
                # reset from the whole dataset,to distill clean data
                cleanset.reset_index(clean_all)
                unlabeledset.reset_index(~clean_all)
                unlabeledset_all = copy.deepcopy(unlabeledset)
                clean_index = np.where(clean_all == 1)[0]
                unlabeled_index = np.where(clean_all == 0)[0]
                closed_set, open_set, clean_set = cleanset.get_sets(sel=True)
        else:
            if epoch == cfg.warmup_epochs:
                step_flagging('after warmup')
            if cleanset is None or unlabeledset is None:
                squential_loader = DataLoader(
                        train_data,
                        batch_size=cfg.batch_size,
                        num_workers=8,
                        drop_last=False,
                        shuffle=False)
                for batch_id, sample in enumerate(squential_loader):
                    (input_w_plus, input_w),targets = sample
                    input_w_plus,input_w = input_w_plus.to(device),input_w.to(device)
                    targets = targets.to(device)
                    
                    input_all = torch.cat([input_w,input_w_plus],dim=0)
                    logits_all = net(input_all,has_open=False)
                    logits_w_plus,logits_w = logits_all.chunk(2)
                    probability = (F.softmax(logits_w_plus, dim=1) + F.softmax(logits_w, dim=1)) / 2
                    C = logits_all.size(1)
                    label_smooth = torch.full(size=(input_w.size(0), C), fill_value=cfg.eps/(C - 1)).to(device)
                    label_smooth.scatter_(dim=1, index=torch.unsqueeze((targets), dim=1), value=1-cfg.eps)
                    prob_clean = 1 - js_div(probability, label_smooth)
                    clean_idx = prob_clean > cfg.tau_clean
                    if batch_id == 0:
                        clean_all = clean_idx
                    else:
                        clean_all = torch.cat([clean_all,clean_idx],0)
                    
                clean_all = variable_to_numpy(clean_all)
                cleanset = copy.deepcopy(train_data)
                unlabeledset = copy.deepcopy(train_data)
                # reset from the whole dataset,to distill clean data
                print('reset cleanset and unlabeled set')
                cleanset.reset_index(clean_all)
                unlabeledset.reset_index(~clean_all)
                unlabeledset_all = copy.deepcopy(unlabeledset)
                clean_index = np.where(clean_all == 1)[0]
                unlabeled_index = np.where(clean_all == 0)[0]
                closed_set, open_set, clean_set = cleanset.get_sets(sel=True)
            unlabeledset = copy.deepcopy(unlabeledset_all)
            if epoch >= cfg.fix_epoch:
                find_id(unlabeledset,net,cfg,device,exclude_known=False,logger=logger)
                unlabeledset.get_sets(sel=True)
            # after warmup
            unlabeledset.control(mode = '110')
            cleanset.control(mode = '011')
            unlabeledset_all.control(mode = '110')
            
            unlabeled_trainloader = DataLoader(unlabeledset,
                                            batch_size = cfg.batch_size * cfg.mu,
                                            num_workers = cfg.num_workers,
                                            drop_last = True)
            unlabeled_trainloader_all = DataLoader(unlabeledset_all,
                                            batch_size=cfg.batch_size * cfg.mu,
                                            num_workers=cfg.num_workers,
                                            drop_last=True)
            labeled_trainloader = DataLoader(cleanset,
                                             batch_size = cfg.batch_size,
                                             num_workers = cfg.num_workers,
                                             drop_last=True)
            
            unlabeled_iter = iter(unlabeled_trainloader)
            unlabeled_all_iter = iter(unlabeled_trainloader_all)
            labeled_iter = iter(labeled_trainloader)
            
            for it in range(0,cfg.steps):
                s = time.time()
                
                optimizer.zero_grad()
                
                try:
                    (inputs_x_s, inputs_x), targets_x = labeled_iter.next()
                except:
                    labeled_iter = iter(labeled_trainloader)
                    (inputs_x_s, inputs_x), targets_x = labeled_iter.next()
                    
                try:
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                except:
                    unlabeled_iter = iter(unlabeled_trainloader)
                    (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                    
                try:
                    (inputs_all_w, inputs_all_s), _ = unlabeled_all_iter.next()
                except:
                    unlabeled_all_iter = iter(unlabeled_trainloader_all)
                    (inputs_all_w, inputs_all_s), _ = unlabeled_all_iter.next()

                b_size = inputs_x.shape[0]

                inputs = torch.cat([inputs_x, inputs_x_s,inputs_all_w, inputs_all_s], 0).to(device)
                targets_x = targets_x.to(device)
                ## Feed data
                logits, logits_open = net(inputs)
                # all unlabeled data
                logits_open_u1, logits_open_u2 = logits_open[2*b_size:].chunk(2)

                ## Loss for labeled samples
                Lx = F.cross_entropy(logits[:2*b_size],
                                        targets_x.repeat(2), reduction='mean')
                L_ova = ova_loss(logits_open[:2*b_size], targets_x.repeat(2))
                L_sup = Lx + L_ova
                ## Open-set entropy minimization
                L_e = ova_ent(logits_open_u1) / 2. + ova_ent(logits_open_u2) / 2.

                ## Soft consistenty regularization
                # (B,2,C)
                logits_open_u1 = logits_open_u1.view(logits_open_u1.size(0), 2, -1)
                logits_open_u2 = logits_open_u2.view(logits_open_u2.size(0), 2, -1)
                probs_open_u1 = F.softmax(logits_open_u1, 1)
                probs_open_u2 = F.softmax(logits_open_u2, 1)
                L_con = torch.mean(torch.sum(torch.sum(torch.abs(
                    probs_open_u1 - probs_open_u2)**2, 1), 1))

                if epoch >= cfg.fix_epoch:
                    inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(device)
                    logits, _ = net(inputs_ws)
                    logits_u_w, logits_u_s = logits.chunk(2)
                    pseudo_label = torch.softmax(logits_u_w.detach()/cfg.T, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(cfg.threshold).float()
                    L_id = (F.cross_entropy(logits_u_s,
                                            targets_u,
                                            reduction='none') * mask).mean()
                    if it == 0:
                        logger.debug(f'selected for fixmatch : {mask.mean().item():.3f}')
                    
                else:
                    L_id = torch.zeros(1).to(device).mean()
                
                # total loss
                loss = L_sup + cfg.lambda_e * L_e  \
                    + cfg.lambda_con * L_con + cfg.lambda_id * L_id
                train_loss.update(loss.item(),inputs.shape[0])
                
                # if cfg.use_fp16:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                loss.backward()
                
                optimizer.step()
                
                scheduler.step()
                
                # ema.update_params(net)
                # ema.apply_shadow(net_ema)
                epoch_train_time.update(time.time() - s, 1)
                
                if (cfg.log_freq is not None and (it + 1) % cfg.log_freq == 0) or (it + 1 == cfg.steps):
                    console_content = f"Epoch:[{epoch + 1:>3d}/{cfg.epochs:>3d}]  " \
                                    f"Iter:[{it + 1:>4d}/{cfg.steps:>4d}]  " \
                                    f"Loss:[{train_loss.avg:4.4f}]  " \
                                    f"{epoch_train_time.avg:6.2f} sec/iter"
                    logger.debug(console_content)

        # evaluate this epoch
        test_acc = evaluate(test_loader, net, device)
        # test_acc_ema = evaluate(test_loader, net_ema, device)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            torch.save(net.state_dict(), f'{result_dir}/best_epoch.pth')
        
        # if test_acc_ema > best_accuracy_ema:
        #     best_accuracy_ema = test_acc_ema
        #     best_epoch_ema = epoch + 1
        #     torch.save(net.state_dict(), f'{result_dir}/best_epoch_ema.pth')

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            # 'best_accuracy_ema': best_accuracy_ema,
            # 'best_epoch_ema': best_epoch_ema,
            # 'ema_state_dict': net_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=f'{result_dir}/checkpoint.pth')

        # logging this epoch
        runtime = time.time() - start_time
        
        
        logger.info(f'epoch: {epoch + 1:>3d} | '
                    f'train loss: {train_loss.avg:>6.4f} | '
                    f'epoch runtime: {runtime:6.2f} sec | '
                    f'net test acc: {test_acc:>6.3f} | '
                    f'best accuracy: {best_accuracy:6.3f} @ epoch: {best_epoch:03d}'
                    # f'net_ema test acc: {test_acc_ema:>6.3f} | '
                    # f'best accuracy ema: {best_accuracy_ema:6.3f} @ epoch: {best_epoch_ema:03d}'
                    )

    # rename results dir -----------------------------------------------------------------------------------------------------------------------------
    os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}')
    # os.rename(result_dir, f'{result_dir}-bestAcc_{best_accuracy:.4f}-bestEmaAcc_{best_accuracy_ema:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--log_prefix', type=str)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--lambda_id', type=float)
    parser.add_argument('--lambda_e', type=float)
    parser.add_argument('--lambda_con', type=float)
    parser.add_argument('--net', type=str)
    parser.add_argument('--lr', type=float)
    
    args = parser.parse_args()

    config = load_from_cfg(args.config)
    override_config_items = [k for k, v in args.__dict__.items() if k != 'config' and v is not None]
    for item in override_config_items:
        config.set_item(item, args.__dict__[item])
    if config.dataset.startswith('cifar'):
        config.log = f'{config.net}-{config.noise_type}_closeset{config.closeset_ratio}_openset{config.openset_ratio}-{config.log_prefix}'
    else:
        config.log = f'{config.net}-{config.log_prefix}'
    print(config)
    return config

def find_id(dataset,model,cfg,device,exclude_known=False,logger = None):
    data_time = AverageMeter()
    end = time.time()
    dataset.control(mode = '001')
    test_loader = DataLoader(
        dataset,
        batch_size = cfg.batch_size,
        num_workers = cfg.num_workers,
        drop_last = False,
        shuffle = False)
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):

            inputs = inputs.to(device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            # (B,2,C)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            known_ind = unk_score < 0.5 
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)
    known_all = variable_to_numpy(known_all)
    if exclude_known:
        ind_selected = np.where(known_all == 0)[0]
    else:
        ind_selected = np.where(known_all != 0)[0]
    data_time.update(time.time() - end)
    logger.debug(f"selected ratio {(len(ind_selected)/ len(known_all))} -- time used: {data_time.avg}" )
    model.train()
    dataset.reset_index(ind_selected)

if __name__ == '__main__':
    params = parse_args()
    print(params.pretrained)
    dev = set_device(params.gpu)
    script_start_time = time.time()
    main(params, dev)
    script_runtime = time.time() - script_start_time
    print(f'Runtime of this script {str(pathlib.Path(__file__))} : {script_runtime:.1f} seconds ({script_runtime/3600:.3f} hours)')
