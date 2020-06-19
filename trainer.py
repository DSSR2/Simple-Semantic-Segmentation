import sys
sys.path.insert(0, '../over9000/')
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import torch.backends.cudnn as cudnn

from my_dataset import SegmentationDataset, provider
from meter import Meter
from tqdm import tqdm
from ranger import Ranger
from lookahead import LookaheadAdam
from loss import BCEDiceJaccardLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dice, iou, f2 = meter.get_metrics()
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | f2_score: %0.4f" % (epoch_loss, iou, dice, f2))
    return dice, iou, f2



class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self, model, path, img_ext, mask_ext, save_path, optim, loss, lr, bs, name, shape=256, crop_type=0):
        self.num_workers = 4
        self.save_path = save_path
        self.batch_size = {"train": bs, "val": 1}
        self.accumulation_steps = bs // self.batch_size['train']
        self.lr = lr
        self.path = path
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.loss = loss
        self.optim = optim
        self.num_epochs = 0
        self.best_val_loss = 1
        self.best_val_dice = 0
        self.best_val_iou = 0
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.name = name
        self.do_cutmix = True
        self.loss_classification = torch.nn.CrossEntropyLoss()
        if self.loss == 'BCE+DICE+IOU':
            self.criterion = BCEDiceJaccardLoss(threshold=None)
        else:
            raise(Exception(f'{self.loss} is not recognized. Please provide a valid loss function.'))

        if self.optim == 'Ranger':
            self.optimizer = Ranger(self.net.parameters(),lr=self.lr)
        elif self.optim == 'LookaheadAdam':
            self.optimizer = LookaheadAdam(self.net.parameters(),lr=self.lr)
        else:
            raise(Exception(f'{self.optim} is not recognized. Please provide a valid optimizer function.'))
            
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, mode="min", patience=4, verbose=True, min_lr = 1e-5)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        
        self.dataloaders = {
            phase: provider(
                phase=phase,
                path = self.path,
                img_ext = self.img_ext,
                mask_ext = self.mask_ext,
                num_workers=0,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.f2_scores = {phase: [] for phase in self.phases}

    def freeze(self):
        for  name, param in self.net.encoder.named_parameters():
            if name.find('bn') != -1:
                param.requires_grad=True
            else:
                param.requires_grad=False
                


    def load_model(self, name, path='models/'):
        state = torch.load(path+name, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print("Loaded model with dice: ", state['best_dice'])
            
    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad=True
     
    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        preds = self.net(images)
        loss = self.criterion(preds, targets)
        return loss, targets, preds

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            if phase == "train" and self.do_cutmix:
                images, targets = self.cutmix(batch, 0.5)
            else:
                images, targets = batch
            seg_loss, outputs, preds = self.forward(images, targets)
            loss = seg_loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(outputs, preds)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou, f2 = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        self.f2_scores[phase].append(f2)
        torch.cuda.empty_cache()
        return epoch_loss, dice, iou

    def train_end(self):
        train_dice = self.dice_scores["train"]
        train_iou = self.iou_scores["train"]
        train_f2 = self.f2_scores["train"]
        train_loss = self.losses["train"]
        
        val_dice = self.dice_scores["val"]
        val_iou = self.iou_scores["val"]
        val_f2 = self.f2_scores["val"]
        val_loss = self.losses["val"]

        df_data=np.array([train_loss, train_dice, train_iou, train_f2, val_loss, val_dice, val_iou, val_f2]).T
        df = pd.DataFrame(df_data,columns = ["train_loss", "train_dice", "train_iou", "train_f2", "val_loss", "val_dice", "val_iou", "val_f2"])
        os.makedirs("./logs/", exist_ok=True)
        df.to_csv('logs/'+self.name+'.csv')

    def fit(self, epochs):
        self.num_epochs+=epochs
        for epoch in range(self.num_epochs-epochs, self.num_epochs):
            self.net.train()
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_val_loss,
                "best_dice": self.best_val_dice,
                "best_iou": self.best_val_iou,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.net.eval()
            with torch.no_grad():
                val_loss, val_dice, val_iou = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_iou > self.best_val_iou:
                print("* New optimal found according to IOU Score!, saving state *")
                state["best_iou"] = self.best_val_iou = val_iou
                torch.save(state, self.save_path+"/"+self.name+'_best.pth')
            torch.save(state, self.save_path+"/"+self.name+'_last.pth')
            print()
            self.train_end()