import cv2
import os
import numpy as np
import albumentations as aug
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose,Cutout, GaussNoise, RandomRotate90, Transpose, RandomBrightnessContrast, RandomCrop)
from albumentations.pytorch import ToTensor
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class SegmentationDataset(Dataset):
    def __init__(self, phase, path=os.getcwd(), img_ext=".jpg", mask_ext=".png", dim=(256,256)):
        '''
        Iniitialize the dataset
        '''
        self.transforms = get_transforms(phase)
        self.phase = phase
        self.path = path
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        train = os.listdir(self.path+'/Images/')
        print("Found ", str(len(train)), " files.")
        train = [i for  i in train if img_ext in i]
        self.train, self.val = train_test_split(train, test_size = 0.2)
        self.dim = dim
    def __getitem__(self, idx):
        '''
        Get one item from dataset
        '''
        if self.phase == 'train':
            img = cv2.imread(self.path+'/Images/'+self.train[idx])
            img = cv2.resize(img, self.dim, interpolation = cv2.INTER_NEAREST)  # Resizing to fit with EfficientNet
            mask = cv2.imread(self.path+'/Masks/' + self.train[idx].split('.')[0]+self.mask_ext, 0)
            mask = cv2.resize(mask, self.dim, interpolation = cv2.INTER_NEAREST)
        elif self.phase == 'val':
            img = cv2.imread(self.path+'/Images/'+self.val[idx])
            img = cv2.resize(img, self.dim, interpolation = cv2.INTER_NEAREST)  # Resizing to fit with EfficientNet
            mask = cv2.imread(self.path+'/Masks/' + self.val[idx].split('.')[0]+self.mask_ext, 0)
            mask = cv2.resize(mask, self.dim, interpolation = cv2.INTER_NEAREST)
        mask = (mask != 0)*255 
        mask = mask.astype(np.uint8)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        return img, mask

    def __len__(self):
        if self.phase == 'train':
            return len(self.train)
        else:
            return len(self.val)

def get_transforms(phase):
    '''
    List of augmentations for training
    '''
    list_transforms = []
    if phase == "train":
        list_transforms.extend([aug.Flip()])
    list_transforms.extend([ToTensor(),])
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(phase, path=os.getcwd(), img_ext=".jpg", mask_ext=".png", batch_size=8, num_workers=0):
    '''
    Returns dataloader for the model training

    usage: dl = provider('train', "./storage/Dataset/", ".png", ".png")
    '''
    image_dataset = SegmentationDataset(phase, path, img_ext, mask_ext)
        
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True,   
    )
    return dataloader