
import os, json
import random
from easydict import EasyDict as edict
from pathlib import Path
from prettyprinter import pprint
from imageio import imread, imsave

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


###################################################################################
import os, sys
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

import copy
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from tqdm import tqdm as tqdm

import logging
logger = logging.getLogger(__name__)

import smp
from models.net_arch import init_model
import wandb

f_score = smp.utils.functional.f_score

# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
diceLoss = smp.utils.losses.DiceLoss(eps=1)
AverageValueMeter =  smp.utils.train.AverageValueMeter

# Augmentations
from dataset.augument import get_training_augmentation, \
    get_validation_augmentation, get_preprocessing

from torch.utils.data import DataLoader
from dataset.wildfire import S1S2 as Dataset # ------------------------------------------------------- Dataset

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
        self.rundir = self.project_dir / self.cfg.experiment.output
        self.model_url = str( self.rundir / "model.pth")

        self.preprocessing_fn = \
            smp.encoders.get_preprocessing_fn(cfg.model.ENCODER, cfg.model.ENCODER_WEIGHTS)

        self.metrics = [smp.utils.metrics.IoU(threshold=0.5),
                        smp.utils.metrics.Fscore()
                    ]

        ''' -------------> need to improve <-----------------'''
        # specify data folder
        self.train_dir = Path(self.cfg.data.dir) / 'train'
        self.valid_dir = Path(self.cfg.data.dir) / 'test'
        '''--------------------------------------------------'''


    def get_dataloaders(self) -> dict:

        """ Data Preparation """
        train_dataset = Dataset(
            self.train_dir, 
            self.cfg, 
            # augmentation=get_training_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.cfg.data.CLASSES,
        )

        valid_dataset = Dataset(
            self.valid_dir, 
            self.cfg, 
            # augmentation=get_validation_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=self.cfg.data.CLASSES,
        )

        train_size = int(len(train_dataset) * self.cfg.model.train_ratio)
        valid_size = len(train_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

        train_loader = DataLoader(train_set, batch_size=self.cfg.model.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(val_set, batch_size=self.cfg.model.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(valid_dataset, batch_size=self.cfg.model.batch_size, shuffle=False, num_workers=4)

        dataloaders = { 
                        'train': train_loader, \
                        'valid': valid_loader, \
                        'test': test_loader, \

                        'train_size': train_size, \
                        'valid_size': valid_size, \
                        'test_size': len(valid_dataset)
                    }

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
        self.history_logs['test'] = []

        # --------------------------------- Train -------------------------------------------
        max_score = self.cfg.model.max_score
        for epoch in range(0, self.cfg.model.max_epoch):
            epoch = epoch + 1
            print(f"\n==> train epoch: {epoch}/{self.cfg.model.max_epoch}")
            self.train_one_epoch(epoch)
            valid_logs = self.valid_logs
            
            # do something (save model, change lr, etc.)
            if valid_logs['iou_score'] > max_score:
                max_score = valid_logs['iou_score']

                if (1 == epoch) or (0 == epoch % self.cfg.model.save_interval):
                    torch.save(self.model, self.model_url)
                    # torch.save(self.model.state_dict(), self.model_url)
                    print('Model saved!')

            # if epoch % 50 == 0:
            #     self.optimizer.param_groups[0]['lr'] = 0.1 * self.optimizer.param_groups[0]['lr']
                        
        
    def train_one_epoch(self, epoch):
        self.model.to(self.DEVICE)
        
        # wandb.
        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            logs = self.step(phase) 
            # print(phase, logs)

            currlr = self.lr_scheduler.get_last_lr()[0] if self.cfg.model.use_lr_scheduler else self.optimizer.param_groups[0]['lr']          
            wandb.log({phase: logs, 'epoch': epoch, 'lr': currlr})

            temp = [logs["total_loss"]] + [logs[self.metrics[i].__name__] for i in range(0, len(self.metrics))]
            self.history_logs[phase].append(temp)

            if phase == 'valid': self.valid_logs = logs



    def step(self, phase) -> dict:
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        # if ('Train' in phase) and (self.cfg.useDataWoCAug):
        #     dataLoader_woCAug = iter(self.dataloaders['Train_woCAug'])

        with tqdm(iter(self.dataloaders[phase]), desc=phase, file=sys.stdout, disable=not self.cfg.model.verbose) as iterator:
            for (x1, x2, y) in iterator:

                # if ('Train' in phase) and (self.cfg.useDataWoCAug):
                #     x0, y0 = next(dataLoader_woCAug)  
                #     x = torch.cat((x0, x), dim=0)
                #     y = torch.cat((y0, y), dim=0)

                x1, x2, y = x1.to(self.DEVICE), x2.to(self.DEVICE), y.to(self.DEVICE)
                self.optimizer.zero_grad()

                y_pred = self.model.forward((x1, x2))

                dice_loss_ =  diceLoss(y_pred, y)
                # focal_loss_ = self.cfg.alpha * focal_loss(y_pred, y)
                # tv_loss_ = 1e-5 * self.cfg.beta * torch.mean(tv_loss(y_pred))

                loss_ = dice_loss_

                # update loss logs
                loss_value = loss_.cpu().detach().numpy()
                loss_meter.add(loss_value)
                # loss_logs = {criterion.__name__: loss_meter.mean}
                loss_logs = {'total_loss': loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)

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
##############################################################



def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(config_path="./config", config_name="siam_unet")
def run_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # wandb.init(config=cfg, project=cfg.project.name, name=cfg.experiment.name)
    wandb.init(config=cfg, project=cfg.project.name, entity=cfg.project.entity, name=cfg.experiment.name)
    # project_dir = Path(hydra.utils.get_original_cwd())
    
    # set randome seed
    set_random_seed(cfg.data.SEED)

    # from experiments.seg_model import SegModel
    mySegModel = SegModel(cfg)
    mySegModel.run()

    # evaluation
    from s1s2_evaluator import evaluate_model
    evaluate_model(cfg, mySegModel.model_url, mySegModel.rundir / "errMap")
    
    wandb.finish()


if __name__ == "__main__":
    run_app()
