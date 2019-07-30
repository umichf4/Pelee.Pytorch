#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import shutil
import argparse
from peleenet import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from data import detection_collate
from configs.CC import Config
from utils.core import *

parser = argparse.ArgumentParser(description='Pelee Training')
parser.add_argument('-c', '--config', default='configs/Pelee_VOC.py')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool,
                    default=True, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()

print_info('----------------------------------------------------------------------\n'
           '|                       Pelee Training Program                       |\n'
           '----------------------------------------------------------------------', ['red', 'bold'])

logger = set_logger(args.tensorboard)
global cfg
cfg = Config.fromfile(args.config)
net = build_net('train', cfg.model.input_size, cfg.model)
optimizer = set_optimizer(net, cfg)
init_net(net, optimizer, cfg, args.resume_net)  # init the network with pretrained
if args.ngpu > 1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benckmark = True

criterion = set_criterion(cfg)
priorbox = PriorBox(anchors(cfg.model))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
  optimizer, mode='min', factor=0.1, patience=1000, verbose=False, 
  threshold=1e-4, threshold_mode='rel', cooldown=100, min_lr=0, eps=1e-08)

with torch.no_grad():
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()


if __name__ == '__main__':
    net.train()
    epoch = args.resume_epoch
    print_info('===> Loading Dataset...', ['red', 'bold'])
    dataset = get_dataloader(cfg, args.dataset, 'train_sets')
    dataset_val = get_dataloader(cfg, args.dataset, 'eval_sets')
    # print(dataset_val._annopath,dataset_val._imgpath,dataset_val.root)
    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = cfg.train_cfg.step_lr[-1] + 1

    stepvalues = cfg.train_cfg.step_lr

    print_info('===> Training STDN on ' + args.dataset, ['red', 'bold'])

    start_iter = args.resume_epoch * epoch_size if args.resume_epoch > 0 else 0
    step_index = 0
    for step in stepvalues:
        if start_iter > step:
            step_index += 1

    batch_iterator_val = iter(data.DataLoader(dataset_val,
                                          batch_size = 8,
                                          shuffle=True,
                                          num_workers=cfg.train_cfg.num_workers,
                                          collate_fn=detection_collate
                                          ))
    images_val, targets_val = batch_iterator_val.next()                                       

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset,
                                                  cfg.train_cfg.per_batch_size * args.ngpu,
                                                  shuffle=True,
                                                  num_workers=cfg.train_cfg.num_workers,
                                                  collate_fn=detection_collate))
            

            if epoch % cfg.model.save_epochs == 0:
                save_checkpoint(net, optimizer, cfg, final=False, datasetname=args.dataset, epoch=epoch)
            epoch += 1
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        # lr = adjust_learning_rate(optimizer, step_index, cfg, args.dataset)
        images, targets = next(batch_iterator)
        
        if cfg.train_cfg.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        if cfg.train_cfg.cuda:
            images_val = images_val.cuda()
            targets_val = [anno.cuda() for anno in targets_val]
        out = net(images)
        out_val = net(images_val)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss_l_val, loss_c_val = criterion(out_val, priors, targets_val)
        loss_val = loss_l_val + loss_c_val
        loss.backward()
        optimizer.step()
        if iteration % epoch_size == 0:
          scheduler.step(loss_val)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        write_logger({'loc_loss': loss_l.item(),
                      'conf_loss': loss_c.item(),
                      'loc_loss_val': loss_l_val.item(),
                      'conf_loss_val': loss_c_val.item(),
                      'loss': loss.item(),
                      'loss_val': loss_val.item(),
                      'lr':lr}, logger, iteration, status=args.tensorboard)
        load_t1 = time.time()
        print_train_log(iteration, cfg.train_cfg.print_epochs,
                        [time.ctime(), epoch, iteration % epoch_size, epoch_size, iteration, load_t1 - load_t0, lr,
                        loss_l.item(), loss_c.item(), loss_l_val.item(), loss_c_val.item()])

    save_checkpoint(net, cfg, final=True,
                    datasetname=args.dataset, epoch=-1)
