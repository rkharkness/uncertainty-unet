import torch
import numpy as np
import matplotlib.pyplot as plt

from unet import VGGNestedUNet, DiceBCELoss
from utils import iou_score, dice2D, bbox, dice_coef

from torchvision.utils import make_grid

from tqdm import tqdm

def check_progress(gt, pred, epoch, phase):
      gt = gt.detach().cpu() #.numpy()
      pred = pred.detach().cpu() #.numpy()
      
      grid_img = make_grid(gt, nrow=4)
      plt.imshow(grid_img.permute(1, 2, 0))
      plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/{phase}_gt_masks{epoch}.png")
     
      grid_img = make_grid(pred, nrow=4)
      plt.imshow(grid_img.permute(1, 2, 0))
      plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/{phase}_pred_masks{epoch}.png")
      

def save_best_model(model, filepath): 
  torch.save(model.state_dict(), filepath)

def validate(model, loss_fn, val_loader, epoch, device='cuda'):
    model.eval()
    val_loss = []
    val_iou = []
    val_dice = []
    count = 0
    with torch.no_grad():
        for inputs, masks in val_loader:
            masks = masks.to(device)
            inputs = inputs.to(device)
            count = count + 1
            if model.deep_supervision_status() == True:
                outputs = model(inputs)
                if count == 5:
                    check_progress(inputs, outputs[-1], epoch, 'val')
                loss = 0
                for output in outputs:
                    loss += loss_fn(output, masks)

                loss /= len(outputs)
                iou = iou_score(outputs[-1], masks)
                dice = dice_coef(outputs[-1], masks)
            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, masks)
                iou = iou_score(outputs, masks)
                dice = dice_coef(outputs, masks)

            val_loss.append(loss.item())
            val_iou.append(iou.item())
            val_dice.append(dice.item())

        val_avg_loss = np.mean(val_loss)
        val_avg_iou = np.mean(val_iou)
        val_avg_dice = np.mean(val_dice)

    model.train()

    return val_avg_loss, val_avg_iou, val_avg_dice

def train(model, train_loader, val_loader, optimizer, loss_fn, k, scheduler, patience=20, num_epochs=200, device='cuda'):
    """Code adapted from: https://github.com/4uiiurz1/pytorch-nested-unet"""
    best_loss = 1e10
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_iou = []
        epoch_dice = []

        progress_bar = tqdm(total=len(train_loader))
        count = 0 
        for inputs, masks in train_loader:
            progress_bar.update(1)

            masks = masks.to(device)
            inputs = inputs.to(device)
            count = count +1
            if model.deep_supervision_status() == True:
                outputs = model(inputs)
                if count == 5:
                    check_progress(inputs, outputs[-1], epoch, 'train')
                
                loss = 0
                for output in outputs:
                    loss += loss_fn(output, masks)

                loss /= len(outputs)
                iou = iou_score(outputs[-1], masks)
                dice = dice_coef(outputs[-1], masks)

            else:
                outputs = model(inputs)
                loss = loss_fn(outputs, masks)
                iou = iou_score(outputs, masks)
                dice = dice_coef(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_iou.append(iou.item())
            epoch_dice.append(dice.item())
            epoch_loss.append(loss.item())

        val_loss, val_iou, val_dice = validate(model, loss_fn, val_loader, epoch)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvements = 0
            save_best_model(model, f'/MULTIX/DATA/HOME/mcd_vgg_nested_unet_cxr_{k}.pth')
            print(f"No improvements for {no_improvements} epochs")

        else:
            no_improvements += 1
            print(f"No improvements for {no_improvements} epochs")
            if patience == no_improvements:
                print("Early stopped !")
                break
		
        print(f"epoch: {epoch} - train loss: {np.mean(epoch_loss)}, train iou: {np.mean(epoch_iou)}, train dice: {np.mean(epoch_dice)}")
        print(f"epoch: {epoch} - val loss: {np.mean(val_loss)}, val iou: {np.mean(val_iou)}, val dice: {np.mean(val_dice)}")

    return model
