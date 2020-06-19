import torch
import numpy as np
from sklearn.metrics import f1_score, fbeta_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import jaccard_similarity_score as jaccard_score
from sklearn.model_selection import train_test_split
class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.threshold = 0.5 
        self.base_dice_scores = []
        self.iou_scores = []
        self.f2_scores = []
        self.phase = phase


    def update(self, y_true, y_preds):
        probs_sig = torch.sigmoid(y_preds)
        iou = soft_jaccard_score(probs_sig, y_true)
        
        y_true = y_true.float().detach().cpu().numpy()
        preds = y_preds.float().detach().cpu().numpy()
        preds = (preds > self.threshold).astype('uint8')
        dice = get_dice(y_true, preds)
        f2 = get_f2(y_true, preds)

        self.base_dice_scores.append(dice)
        self.f2_scores.append(f2)
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        f2 = np.mean(self.f2_scores)
        iou = np.nanmean(self.iou_scores)
        return dice, iou, f2

def get_dice(y_true, y_pred):
    dice = 0
    batch_size = y_true.shape[0]
    channel_size = y_true.shape[1]
    for i in range(batch_size):
      for j in range(channel_size):
        y_tr = y_true[i][j]
        y_pr = y_pred[i][j]
        y_tr = y_tr.reshape(-1,)
        y_pr = y_pr.reshape(-1,)
        dice += f1_score(y_tr, y_pr)/(batch_size*channel_size)
    return dice

def get_f2(y_true, y_pred):
    f2 = 0
    batch_size = y_true.shape[0]
    channel_size = y_true.shape[1]
    for i in range(batch_size):
      for j in range(channel_size):
        y_tr = y_true[i][j]
        y_pr = y_pred[i][j]
        y_tr = y_tr.reshape(-1,)
        y_pr = y_pr.reshape(-1,)
        f2 += fbeta_score(y_tr, y_pr, beta=0.5)/(batch_size*channel_size)
    return f2

def soft_jaccard_score(y_pred: torch.Tensor, y_true: torch.Tensor, smooth=0.0, eps=1e-7, threshold=0.6) -> torch.Tensor:
    assert y_pred.size() == y_true.size()
    bs = y_true.size(0)
    num_classes = y_pred.size(1)
    dims = (0, 2)
    y_pred = (y_pred>threshold).float()
    y_true = y_true.view(bs, num_classes, -1)
    y_pred = y_pred.view(bs, num_classes, -1)
    
    if dims is not None:
        intersection = torch.sum(y_pred * y_true, dim=dims)
        cardinality = torch.sum(y_pred + y_true, dim=dims)
    else:
        intersection = torch.sum(y_pred * y_true)
        cardinality = torch.sum(y_pred + y_true)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union.clamp_min(eps) + smooth)
    return jaccard_score.mean().item()