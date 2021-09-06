import os, sys, time, copy
import numpy as np
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm as tqdm
from easydict import EasyDict as edict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import smp
from smp.losses.focal import FocalLoss
from smp.losses.dice import DiceLoss

from models.net_arch import init_model
import wandb

focal_loss = FocalLoss(mode="multiclass")
dice_loss = DiceLoss(mode="multiclass", smooth=1)

AverageValueMeter =  smp.utils.train.AverageValueMeter

# dataset
from torch.utils.data import DataLoader
# from dataset.camvid import Dataset
from dataset.mtbs_l8 import Dataset

# Augmentations
from dataset.augument import get_training_augmentation, \
    get_validation_augmentation, get_preprocessing

from models.lr_schedule import get_cosine_schedule_with_warmup


def format_logs(logs):
    str_logs = ['{}: {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s

class SegModel(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.project_dir = Path(hydra.utils.get_original_cwd())

        self.cfg = cfg
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = init_model(cfg)
        # self.model_url = str(self.project_dir / "outputs" / "best_model.pth")
        self.model_url = str(self.project_dir / "outputs" / os.path.split(self.cfg.experiment.output)[-1] / "best_model.pth")

        self.preprocessing_fn = \
            smp.encoders.get_preprocessing_fn(cfg.model.ENCODER, cfg.model.ENCODER_WEIGHTS)

        self.metrics = [smp.utils.metrics.IoU(threshold=0.5),
                        smp.utils.metrics.Fscore()]

        # specify data folder
        # data_dir = self.project_dir / "data" / "CamVid"
        data_dir = self.project_dir / "data" / "MTBS_L8_Tiles"
        
        self.train_dir = data_dir / 'US'
        self.valid_dir = data_dir / 'US'
        self.test_dir = data_dir /  'train'


    def get_dataloaders(self) -> dict:

        """ Data Preparation """
        train_dataset = Dataset(
            self.train_dir, 
            None, 
            # augmentation=get_training_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.cfg.data.CLASSES,
        )

        valid_dataset = Dataset(
            self.valid_dir, 
            None, 
            # augmentation=get_validation_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.cfg.data.CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.model.batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=self.cfg.model.batch_size, shuffle=False, num_workers=4)

        dataloaders = { 'train': train_loader, \
                        'valid': valid_loader, \
                        'train_size': len(train_dataset),
                        'valid_size': len(valid_dataset)}

        return dataloaders


    def run(self) -> None:

        self.dataloaders = self.get_dataloaders()
        self.optimizer = torch.optim.Adam([dict(
                params=self.model.parameters(), 
                lr=self.cfg.model.learning_rate, 
                weight_decay=self.cfg.model.weight_decay)])

        # lr scheduler
        per_epoch_steps = self.dataloaders['train_size'] // self.cfg.model.batch_size
        total_training_steps = self.cfg.model.max_epoch * per_epoch_steps
        warmup_steps = self.cfg.model.warmup_coef * per_epoch_steps
        if self.cfg.model.use_lr_scheduler:
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_training_steps)

        self.history_logs = edict()
        self.history_logs['train'] = []
        self.history_logs['valid'] = []

        # --------------------------------- Train -------------------------------------------
        for epoch in range(0, self.cfg.model.max_epoch):
            epoch = epoch + 1
            print(f"\n==> train epoch: {epoch}/{self.cfg.model.max_epoch}")
            valid_logs = self.train_one_epoch(epoch)
            
            torch.save(self.model, self.model_url)
            if False:
                # do something (save model, change lr, etc.)
                if valid_logs['iou_score'] > self.cfg.model.max_score:
                    max_score = valid_logs['iou_score']
                    torch.save(self.model, self.model_url)
                    # torch.save(self.model.state_dict(), self.model_url)
                    print('Model saved!')

            if epoch % 50 == 0:
                self.optimizer.param_groups[0]['lr'] = 0.1 * self.optimizer.param_groups[0]['lr']
                        
    def train_one_epoch(self, epoch):
        self.model.to(self.DEVICE)
        
        # wandb.
        for phase in ['train', 'valid']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            logs = self.step(phase) 

            currlr = self.lr_scheduler.get_last_lr()[0] if self.cfg.model.use_lr_scheduler else self.optimizer.param_groups[0]['lr']          
            wandb.log({phase: logs, 'epoch': epoch, 'lr': currlr})

            # temp = [logs["total_loss"]] + [logs[self.metrics[i].__name__] for i in range(0, len(self.metrics))]
            # self.history_logs[phase].append(temp)

            if phase == 'valid':
                valid_logs = logs
                return valid_logs


    def step(self, phase) -> dict:
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {f"{metric.__name__}_{cls}": AverageValueMeter() for cls in range(6) for metric in self.metrics}

        if ('Train' in phase) and (self.cfg.useDataWoCAug):
            dataLoader_woCAug = iter(self.dataloaders['Train_woCAug'])

        with tqdm(iter(self.dataloaders[phase]), desc=phase, file=sys.stdout, disable=not self.cfg.model.verbose) as iterator:
            for (x, y) in iterator:
                if ('Train' in phase) and (self.cfg.useDataWoCAug):
                    x0, y0 = next(dataLoader_woCAug)  
                    x = torch.cat((x0, x), dim=0)
                    y = torch.cat((y0, y), dim=0)

                x, y = x.to(self.DEVICE), y.to(self.DEVICE)
                self.optimizer.zero_grad()

                y_pred = self.model.forward(x)

                dice_loss_value =  dice_loss(y_pred, y.long())
                focal_loss_value = focal_loss(y_pred, y.long())

                loss_ = dice_loss_value + focal_loss_value

                # update loss logs
                loss_value = loss_.cpu().detach().numpy()
                loss_meter.add(loss_value)
                # loss_logs = {criterion.__name__: loss_meter.mean}
                loss_logs = {'total_loss': loss_meter.mean}
                logs.update(loss_logs)

                label_pred = torch.argmax(y_pred, axis=1).squeeze()

                for cls in range(6):
                    cls_mask_true = (y == cls).type(torch.float)
                    cls_mask_pred = (label_pred == cls).type(torch.float)
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(cls_mask_pred, cls_mask_true).cpu().detach().numpy()
                        metrics_meters[f"{metric_fn.__name__}_{cls}"].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                # print(logs)

                if self.cfg.model.verbose:
                    s = format_logs(logs)
                    iterator.set_postfix_str(s)

                if phase == 'train':
                    loss_.backward()
                    self.optimizer.step()

                    if self.cfg.model.use_lr_scheduler:
                        self.lr_scheduler.step()

            return logs