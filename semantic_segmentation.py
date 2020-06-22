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
from glob import glob
from sklearn.metrics import f1_score, fbeta_score
#For Transformations
import cv2
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

'''
Semantic Segmentation Class
'''
class semantic_segmentation():
    def __init__(self, model="S0", path="./Data/", img_ext=".jpg", mask_ext=".png", save_path=os.getcwd()+"/models/", img_size=(256,256)):
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
        if(os.path.isdir(self.model_name)):
            try:
                f = open(self.model_name+"/Model.txt")
                name = f.read()
                f.close()
                checkpoint = torch.load(self.model_name+"/"+name+"_best.pth")
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model_name = name
                print("Model Loaded Successfully!")
            except:
                print("Unable to load model. Try again.")


    def train(self, epochs, lr=1e-3, bs=4):
        print("Training Started!")
        model_trainer = Trainer(model = self.model, path=self.path, img_ext=self.img_ext, mask_ext=self.mask_ext, save_path=self.save_path, optim = "Ranger", loss = "BCE+DICE+IOU", lr = lr, bs = bs, name = self.model_name, shape=self.img_size[0])
        model_trainer.do_cutmix = False
        model_trainer.freeze()
        model_trainer.fit(epochs//3)
        model_trainer.do_cutmix = False
        model_trainer.unfreeze()
        model_trainer.fit(epochs//3)
        model_trainer.do_cutmix = False
        model_trainer.freeze()
        model_trainer.fit(epochs//4)
        file_name = self.save_path+"/model.txt"
        with open(file_name, 'w') as f:
            f.write(self.model_name)
            f.close()
        print("Training complete!")

    def get_mask(self, img, thresh):
        img = Compose([ToTensor()])(image = img)["image"]
        img = img.unsqueeze(0)
        with torch.no_grad():
            y_preds = self.model(img.type('torch.FloatTensor'))
        y_preds = nn.Sigmoid()(y_preds)
        y_preds = y_preds[0].squeeze(0).cpu().data.numpy()
        y_preds = (y_preds > thresh).astype('uint8')*255
        return y_preds

    def do_predict(self, img, fname, thresh):
        image = img
        y_preds = get_mask(img, thresh)
        image = image.astype('uint8')
        zeros = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        zeros[:,:,2] = y_preds
        mask = zeros.astype("uint8")
        dst = cv2.addWeighted(image,0.7,mask,0.3,0)
        cv2.imwrite(self.op_path+"/"+fname, dst)
    
    def evaluate(self, path):
        files = glob(path+"/Images/*.png")+glob(path+"/Images/*.jpg")+glob(path+"/Images/*.bmp")
        scores = {}
        
        df_rows = []
        for i in tqdm(files):
            fname = i.split("\\")[-1].split(".")[0]
            img = cv2.imread(i)
            img = cv2.resize(img, self.img_size, interpolation = cv2.INTER_NEAREST)
            mask = cv2.imread(path+"/Masks/"+fname+self.mask_ext, 0)
            mask = cv2.resize(mask, self.img_size, interpolation = cv2.INTER_NEAREST)
            pred = self.get_mask(img, 0.4)
            scores = get_scores(mask, pred)
            df_rows.append([fname, scores[0], scores[1], scores[2]])
        df = pd.DataFrame(df_rows, columns=["Filename", "IoU", "Dice", "F2"])

        df.to_csv("Validation.csv", index=False)
        f2_mean = df["F2"].mean()
        print("IoU Mean \tIoU Max \tIoU Min ")
        print(df["IoU"].mean(), "\t", df["IoU"].max(), "\t", df["IoU"].min())
        print()

        print("Dice Mean \tDice Max \tDice Min")
        print(df["Dice"].mean(), "\t", df["Dice"].max(), "\t", df["Dice"].min())
        print()

        print("F2 Mean \tF2 Max \tF2 Min")
        print(df["F2"].mean(), "\t", df["F2"].max(), "\t", df["F2"].min())
        return 0

    def predict(self, test_path, op_path, thresh=0.4):
        print("Prediction Started!")
        self.op_path = op_path
        os.makedirs(op_path, exist_ok=True)
        checkpoint = torch.load(self.save_path+"/"+self.model_name+"_best.pth")
        self.model.load_state_dict(checkpoint["state_dict"])
        sing_f = False
        shape = self.img_size
        if(os.path.isdir(test_path)):
            print("Loading files from ", test_path)
            files = glob(test_path+"/*.png")+glob(test_path+"/*.jpg")+glob(test_path+"/*.bmp")
        else:
            files = [test_path]
            sing_f = True
        print("Predicting on ", len(files), " files.")
        for i in tqdm(files):
            fname = i.split("\\")[-1]
            img = cv2.imread(i)
            img = cv2.resize(img, shape, interpolation = cv2.INTER_NEAREST)
            self.do_predict(img, fname, thresh)
            if(sing_f):
                print(fname + " output saved in "+op_path+"/"+fname)
            

def get_scores(y_true, y_pred):
    y_true = y_true.astype("bool")
    y_pred = y_pred.astype("bool")
    dice = 0
    f2 = 0
    overlap = y_true*y_pred 
    union = y_true + y_pred 
    IoU = overlap.sum()/float(union.sum())
    y_tr = y_true.reshape(-1,)
    y_pr = y_pred.reshape(-1,)
    dice += f1_score(y_tr, y_pr)
    f2 += fbeta_score(y_tr, y_pr, beta=0.5)
    return IoU, dice, f2

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

    elif(os.path.isdir(model_name)):
        try:
            f = open(model_name+"/Model.txt")
            name = f.read()
            f.close()
            print("Loading pretrained model")
            return get_model(name)
        except:
            print("Folder Error, loading S0")
            return get_model("S0")
    else:
        print("Unknown model ", model_name, ". Loading S0")
        return smp.Unet("efficientnet-b0", encoder_weights='imagenet', classes=1, activation=None)