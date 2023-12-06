import torch
import numpy as np

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target, sigmoid=True):
    smooth = 1e-5
    # output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    if sigmoid == True:
      output = torch.sigmoid(output).view(-1)
    else:
      output = output.view(-1)
      
    target = target.view(-1)
    # target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def dice_coef_loss(output, target):
	return 1. - dice_coef(output, target)

# component analysis - open cv https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
def iou_bin_(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union

    
def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def rec(y_true, y_pred):
    # flatten the image arrays for true and pred
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    return (torch.sum(y_true * y_pred)/ (torch.sum(y_true) + torch.tensor(1e-10)))

def prec(y_true, y_pred):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    return (torch.sum(y_true * y_pred) / (torch.sum(y_pred) + torch.tensor(1e-10)))

def iou_(y_true, y_pred):  #this can be used as a loss if you make it negative
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    union = y_true + ((1 - y_true) * y_pred)
    return (torch.sum(y_true * y_pred) + torch.tensor(1e-10)) / (torch.sum(union, axis=-1) + torch.tensor(1e-10))

def iou_loss_(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred), axis=-1)
    union = torch.sum(y_true,-1) + torch.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou