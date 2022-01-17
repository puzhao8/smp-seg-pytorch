# checkpoint: https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

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
from smp.base.modules import Activation
from models.model_selection import get_model
import wandb

# f_score = smp.utils.functional.f_score
# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
# diceLoss = smp.utils.losses.DiceLoss(eps=1)
from models.loss_ref import soft_dice_loss, soft_dice_loss_balanced, jaccard_like_loss, jaccard_like_balanced_loss

AverageValueMeter =  smp.utils.train.AverageValueMeter

# Augmentations
from dataset.augument import get_training_augmentation, \
    get_validation_augmentation, get_preprocessing

from torch.utils.data import DataLoader
from dataset.wildfire import S1S2 as Dataset # ------------------------------------------------------- Dataset

from models.lr_schedule import get_cosine_schedule_with_warmup, PolynomialLRDecay


def format_logs(logs):
    str_logs = ['{}: {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s


def loss_fun(CFG, DEVICE='cuda'):
    if CFG.MODEL.LOSS_TYPE == 'BCELoss':
        criterion = nn.BCELoss()

    elif CFG.MODEL.LOSS_TYPE == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss() # includes sigmoid activation

    elif CFG.MODEL.LOSS_TYPE == 'DiceLoss':
        criterion = smp.utils.losses.DiceLoss(eps=1, activation=CFG.MODEL.ACTIVATION)

    elif CFG.MODEL.LOSS_TYPE == 'CrossEntropyLoss':
        balance_weight = [CFG.MODEL.NEGATIVE_WEIGHT, CFG.MODEL.POSITIVE_WEIGHT]
        balance_weight = torch.tensor(balance_weight).float().to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight = balance_weight)
        
    elif CFG.MODEL.LOSS_TYPE == 'SoftDiceLoss':
        criterion = soft_dice_loss 
    elif CFG.MODEL.LOSS_TYPE == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced
    elif CFG.MODEL.LOSS_TYPE == 'JaccardLikeLoss':
        criterion = jaccard_like_loss
    elif CFG.MODEL.LOSS_TYPE == 'ComboLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + soft_dice_loss(pred, gts)
    elif CFG.MODEL.LOSS_TYPE == 'WeightedComboLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + 10 * soft_dice_loss(pred, gts)
    elif CFG.MODEL.LOSS_TYPE == 'FrankensteinLoss':
        criterion = lambda pred, gts: F.binary_cross_entropy_with_logits(pred, gts) + jaccard_like_balanced_loss(pred, gts)

    return criterion

class SegModel(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.PROJECT_DIR = Path(hydra.utils.get_original_cwd())

        self.cfg = cfg
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.DEVICE = 'cpu'

        self.model = get_model(cfg)
        self.activation = Activation(self.cfg.MODEL.ACTIVATION)

        # self.MODEL_URL = str(self.PROJECT_DIR / "outputs" / "best_model.pth")
        self.RUN_DIR = self.PROJECT_DIR / self.cfg.EXP.OUTPUT
        self.MODEL_URL = str(self.RUN_DIR / "model.pth")

        # if CFG.MODEL.ENCODER is not None:
        #     self.preprocessing_fn = \
        #         smp.encoders.get_preprocessing_fn(CFG.MODEL.ENCODER, CFG.MODEL.ENCODER_WEIGHTS)

        self.metrics = [smp.utils.metrics.IoU(threshold=0.5, activation=None),
                        smp.utils.metrics.Fscore(activation=None)
                    ]

        '''--------------> need to improve <-----------------'''
        # specify data folder
        self.TRAIN_DIR = Path(self.cfg.DATA.DIR) / 'train'
        self.VALID_DIR = Path(self.cfg.DATA.DIR) / 'test'
        '''--------------------------------------------------'''
    
    def get_dataloaders(self) -> dict:

        if self.cfg.MODEL.NUM_CLASSES == 1:
            classes = ['burned']
        elif self.cfg.MODEL.NUM_CLASSES == 2:
            classes = ['unburn', 'burned']
        elif self.cfg.MODEL.NUM_CLASSES > 2:
            print(" ONLY ALLOW ONE or TWO CLASSES SO FAR !!!")
            pass

        """ Data Preparation """
        train_dataset = Dataset(
            self.TRAIN_DIR, 
            self.cfg, 
            # augmentation=get_training_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=classes,
        )

        valid_dataset = Dataset(
            self.VALID_DIR, 
            self.cfg, 
            # augmentation=get_validation_augmentation(), 
            # preprocessing=get_preprocessing(self.preprocessing_fn),
            classes=classes,
        )

        generator=torch.Generator().manual_seed(self.cfg.RAND.SEED)
        train_size = int(len(train_dataset) * self.cfg.DATA.TRAIN_RATIO)
        valid_size = len(train_dataset) - train_size
        train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, valid_size], generator=generator)

        train_loader = DataLoader(train_set, batch_size=self.cfg.MODEL.BATCH_SIZE, shuffle=True, num_workers=4, generator=generator)
        valid_loader = DataLoader(val_set, batch_size=self.cfg.MODEL.BATCH_SIZE, shuffle=True, num_workers=4, generator=generator)
        test_loader = DataLoader(valid_dataset, batch_size=self.cfg.MODEL.BATCH_SIZE, shuffle=True, num_workers=4, generator=generator)

# means = []
# stds = []
# for img in list(iter(train_loader)):
#     print(img.shape)
#     means.append(torch.mean(img))
#     stds.append(torch.std(img))

# mean = torch.mean(torch.tensor(means))
# std = torch.mean(torch.tensor(stds))
        
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
        self.model.to(self.DEVICE)
        self.criterion = loss_fun(self.cfg)

        self.dataloaders = self.get_dataloaders()
        self.optimizer = torch.optim.Adam([dict(
                params=self.model.parameters(), 
                lr=self.cfg.MODEL.LEARNING_RATE, 
                weight_decay=self.cfg.MODEL.WEIGHT_DECAY)])

        """ ===================== >> learning rate scheduler << ========================= """
        per_epoch_steps = self.dataloaders['train_size'] // self.cfg.MODEL.BATCH_SIZE
        total_training_steps = self.cfg.MODEL.MAX_EPOCH * per_epoch_steps

        self.USE_LR_SCHEDULER = True \
            if self.cfg.MODEL.LR_SCHEDULER in ['cosine_warmup', 'polynomial'] \
            else True

        if self.cfg.MODEL.LR_SCHEDULER == 'cosine':
            ''' cosine scheduler '''
            warmup_steps = self.cfg.MODEL.COSINE_SCHEDULER.WARMUP * per_epoch_steps
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_training_steps)

        elif self.cfg.MODEL.LR_SCHEDULER == 'poly':
            ''' polynomial '''
            self.lr_scheduler = PolynomialLRDecay(self.optimizer, 
                            max_decay_steps=total_training_steps, 
                            end_learning_rate=self.cfg.MODEL.POLY_SCHEDULER.END_LR, #1e-5, 
                            power=self.cfg.MODEL.POLY_SCHEDULER.POWER, #0.9
                        )
        else:
            pass


        # self.history_logs = edict()
        # self.history_logs['train'] = []
        # self.history_logs['valid'] = []
        # self.history_logs['test'] = []

        # --------------------------------- Train -------------------------------------------
        max_score = self.cfg.MODEL.MAX_SCORE
        self.iters = 0
        for epoch in range(0, self.cfg.MODEL.MAX_EPOCH):
            epoch = epoch + 1
            print(f"\n==> train epoch: {epoch}/{self.cfg.MODEL.MAX_EPOCH}")
            self.train_one_epoch(epoch)
            valid_logs = self.valid_logs
            
            # do something (save model, change lr, etc.)
            if valid_logs['iou_score'] > max_score:
                max_score = valid_logs['iou_score']

                if (1 == epoch) or (0 == (epoch % self.cfg.MODEL.SAVE_INTERVAL)):
                    torch.save(self.model, self.MODEL_URL)
                    # torch.save(self.model.state_dict(), self.MODEL_URL)
                    print('Model saved!')

            # if epoch % 50 == 0:
            #     self.optimizer.param_groups[0]['lr'] = 0.1 * self.optimizer.param_groups[0]['lr']
                        
        
    def train_one_epoch(self, epoch):
    
        # wandb.
        for phase in ['train', 'valid', 'test']:
            if phase == 'train':
                self.model.train()
            else:
                self.model.eval()

            logs = self.step(phase) 
            # print(phase, logs)

            currlr = self.optimizer.param_groups[0]['lr'] 
            wandb.log({phase: logs, 'epoch': epoch, 'lr': currlr})

            # temp = [logs["total_loss"]] + [logs[self.metrics[i].__name__] for i in range(0, len(self.metrics))]
            # self.history_logs[phase].append(temp)

            if phase == 'valid': self.valid_logs = logs


    def step(self, phase) -> dict:
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        # if ('Train' in phase) and (self.cfg.useDataWoCAug):
        #     dataLoader_woCAug = iter(self.dataloaders['Train_woCAug'])

        with tqdm(iter(self.dataloaders[phase]), desc=phase, file=sys.stdout, disable=not self.cfg.MODEL.VERBOSE) as iterator:
            for i, (x, y) in enumerate(iterator):
                self.optimizer.zero_grad()

                ''' move data to GPU '''
                input = []
                for x_ in x: input.append(x_.to(self.DEVICE))
                y = y.to(self.DEVICE)
                # print(len(input))

                ''' do prediction '''
                if 'UNet_resnet' in self.cfg.MODEL.ARCH: 
                    input = input[0]
                    out = self.model.forward(input)
                else:
                    out = self.model.forward(input)[-1]
                y_pred = self.activation(out) # If use this, set IoU/F1 metrics activation=None

                ''' compute loss '''
                loss_ = self.criterion(out, y)

                ''' Back Propergation (BP) '''
                if phase == 'train':
                    loss_.backward()
                    self.optimizer.step()
                    self.iters = self.iters + 1

                    ''' Iteration-Wise log for train stage only '''
                    if self.cfg.MODEL.STEP_WISE_LOG:
                        self.iters = self.iters + 1
                        currlr = self.optimizer.param_groups[0]['lr'] 
                        # wandb.log({'x0.mean': x[0].mean()})
                        wandb.log({phase: logs, 'iters': self.iters, 'lr': currlr})

                    if self.USE_LR_SCHEDULER:
                        self.lr_scheduler.step()

                # if mask is in one-hot: NCWH, C=NUM_CLASSES (C>1), and do nothing if C=1
                if y.shape[1] >= 2: 
                    y = self.activation(y)

                ''' update loss and metrics logs '''
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
                # print(self.iters, x[0].mean().item(), loss_.item())

                if self.cfg.MODEL.VERBOSE:
                    s = format_logs(logs)
                    iterator.set_postfix_str(s)

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


@hydra.main(config_path="./config", config_name="unet")
def run_app(cfg : DictConfig) -> None:

    ''' set randome seed '''
    os.environ['HYDRA_FULL_ERROR'] = str(1)
    os.environ['PYTHONHASHSEED'] = str(cfg.RAND.SEED) #cfg.RAND.SEED
    if cfg.RAND.DETERMIN:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=":4096:8" #https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        torch.use_deterministic_algorithms(True)
    set_random_seed(cfg.RAND.SEED, deterministic=cfg.RAND.DETERMIN)

    # wandb.init(config=cfg, project=cfg.project.name, name=cfg.EXP.name)
    import pandas as pd
    from prettyprinter import pprint
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_flat = pd.json_normalize(cfg_dict, sep='.').to_dict(orient='records')[0]

    # cfg.MODEL.DEBUG = False
    # if not cfg.MODEL.DEBUG:
    wandb.init(config=cfg_flat, project=cfg.PROJECT.NAME, entity=cfg.PROJECT.ENTITY, name=cfg.EXP.NAME)
    pprint(cfg_flat)

    ''' train '''
    # from experiments.seg_model import SegModel
    mySegModel = SegModel(cfg)
    mySegModel.run()

    ''' inference '''
    from s1s2_evaluator import evaluate_model
    evaluate_model(cfg, mySegModel.MODEL_URL, mySegModel.RUN_DIR / "errMap")

    ''' compute IoU and F1 for all events '''
    from utils.iou4all import compute_IoU_F1
    compute_IoU_F1(phase="test_images", 
                    result_dir=mySegModel.RUN_DIR / "errMap", 
                    dataset_dir=cfg.DATA.DIR)
    
    # if not cfg.MODEL.DEBUG:
    wandb.finish()


if __name__ == "__main__":

    run_app()
