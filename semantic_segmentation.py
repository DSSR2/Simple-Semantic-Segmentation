import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import time
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import torch.optim as optim
import random
import sys
import glob
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from my_dataset import get_transforms
from trainer import Trainer
from tqdm import tqdm
#For Transformations
import cv2
import glob
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, sampler
import albumentations as aug
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose,Cutout, GaussNoise, RandomRotate90, Transpose, RandomBrightnessContrast, RandomCrop)
from albumentations.pytorch import ToTensor


'''
Initialisation
'''
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(69)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
class semantic_segmentation():
    def __init__(self, path, model, img_ext=".jpg", mask_ext=".png", save_path=os.getcwd()+"/models/", img_size=(256,256)):
        self.path = path
        self.image_path = path + "/Images/"
        self.mask_path = path + "/Masks/"
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.model_name = model
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.img_size = img_size
        self.model = get_model(self.model_name)
    
    def train(self, epochs, lr=1e-3, bs=4):
        model_trainer = Trainer(model = self.model, path=self.path, img_ext=self.img_ext, mask_ext=self.mask_ext, optim = "Ranger", loss = "BCE+DICE+IOU", lr = lr, bs = bs, name = self.model_name, shape=self.img_size[0])
        model_trainer.do_cutmix = False
        model_trainer.freeze()
        model_trainer.fit(epochs//3)
        model_trainer.do_cutmix = False
        model_trainer.unfreeze()
        model_trainer.fit(epochs//3)
        model_trainer.do_cutmix = False
        model_trainer.freeze()
        model_trainer.fit(epochs//4)

    def do_predict(self, img, fname, thresh):
        image = img
        img = Compose([ToTensor()])(image = img)["image"]
        img = img.unsqueeze(0)
        if(torch.cuda.is_available):
            y_preds = model(img.type('torch.cuda.FloatTensor'))
        else:
            y_preds = model(img.type('torch.FloatTensor'))
        y_preds = nn.Sigmoid()(y_preds)
        y_preds = y_preds[0].squeeze(0).detach().cpu().numpy()
        y_preds = (y_preds > thresh).astype('uint8')*255
        image = image.astype('uint8')
        zeros = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        zeros[:,:,2] = y_preds
        mask = zeros.astype("uint8")
        dst = cv2.addWeighted(image,0.7,mask,0.3,0)
        cv2.imwrite(self.op_path+"/"+fname, dst)

    def predict(self, test_path, op_path, thresh=0.4):
        self.op_path = op_path
        os.makedirs(op_path, exist_ok=True)
        checkpoint = torch.load(self.save_path+"/"+self.model_name+"_best.pth")
        self.model.load_state_dict(checkpoint["state_dict"])
        sing_f = False
        shape = self.img_size
        if(os.path.isdir(path)):
            files = glob(path+"/*.png")+glob(path+"/*.jpg")+glob(path+"/*.bmp")
        else:
            files = [path]
            sing_f = True

        for i in tqdm(files):
            fname = i.split("\\")[-1]
            img = cv2.imread(i)
            img = cv2.resize(img, shape, interpolation = cv2.INTER_NEAREST)
            do_predict(img, fname, thresh)
            print(fname + " output saved in "+op_path+"/"+fname)
            


def get_model(model_name):
    if(model_name == "S0"):
        return smp.Unet("efficientnet-b0", encoder_weights='imagenet', classes=1, activation=None)
    elif(model_name == "S1"):
        return smp.Unet("efficientnet-b1", encoder_weights='imagenet', classes=1, activation=None)
    elif(model_name == "S2"):
        return smp.Unet("efficientnet-b2", encoder_weights='imagenet', classes=1, activation=None)
    elif(model_name == "S3"):
        return smp.Unet("efficientnet-b3", encoder_weights='imagenet', classes=1, activation=None)
    elif(model_name == "S4"):
        return smp.Unet("efficientnet-b4", encoder_weights='imagenet', classes=1, activation=None)
    elif(model_name == "S5"):
        return smp.Unet("efficientnet-b5", encoder_weights='imagenet', classes=1, activation=None)
    else:
        print("Unknown model ", model_name, ". Loading S0")
        return smp.Unet("efficientnet-b0", encoder_weights='imagenet', classes=1, activation=None)






# def plot(img, model, thresh):
#     image = img
#     augmented = get_transforms('test')(image = img)
#     img = augmented['image']
#     img = img.unsqueeze(0)
#     y_preds = model(img.type('torch.cuda.FloatTensor'))
#     y_preds = nn.Sigmoid()(y_preds)
#     y_preds = y_preds[0].squeeze(0).detach().cpu().numpy()
#     y_preds = (y_preds > thresh).astype('uint8')*255
#     res = np.hstack([image[:, :, 0], y_preds])
#     plt.imshow(res)
#     plt.show()
    
# def predict(model_path, img_path = None, folder_path = None, thresh = 0.4):
#     print(model_path)
#     checkpoint = torch.load(model_path)
#     m.load_state_dict(checkpoint["state_dict"])
#     shape = (256, 256)
#     if folder_path is not None:
#         path = os.listdir(folder_path)
#         for i in range(len(path)):
#             img = cv2.imread(os.path.join(folder_path, path[i]))
#             img = cv2.resize(img, shape, interpolation = cv2.INTER_NEAREST)
#             plot(img, m, thresh)

#     elif img_path is not None:               
#         img = cv2.imread(img_path[:-3])
#         img = cv2.resize(img, shape, interpolation = cv2.INTER_NEAREST)
#         plot(img, m, thresh)