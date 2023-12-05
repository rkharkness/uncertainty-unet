import pandas as pd
import numpy as np

import torch
from sklearn.model_selection import KFold
import torch.optim as optim
from torch.nn import BCELoss

import matplotlib.pyplot as plt

from dataloaders import CustomDataLoader, create_dataloader
from unet import VGGNestedUNet, DiceBCELoss
from training import train

from torchvision.utils import make_grid

if __name__ == "__main__":
    
    full_data = pd.read_csv("/MULTIX/DATA/HOME/COVID_QU_Ex/covid_qu_ex.csv")

    train_df = full_data[full_data['split'] == 'Train']
    val_df = full_data[full_data['split'] == 'Val']

    train_df = pd.concat([train_df, val_df])
    train_df=train_df.reset_index(drop=True)
    
    seed = 0
    np.random.seed(seed)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    fold_no = 1
    for train_idx, val_idx in kfold.split(train_df):
      train_data = train_df.iloc[train_idx]
      train_data = train_data.reset_index(drop=True)
      
      val_data = train_df.iloc[val_idx]
      val_data = val_data.reset_index(drop=True)
      
      train_loader = create_dataloader(bs=4, custom_dataloader=CustomDataLoader, dataframe=train_data, train=True, num_workers=8)
      
      device = torch.device('cuda')
      model = VGGNestedUNet(num_classes=1, deep_supervision_status=True)
      model = model.to(device)
      
      batch_x, batch_y = next(iter(train_loader))
      grid_img = make_grid(batch_x, nrow=4)
      plt.imshow(grid_img.permute(1, 2, 0))
      plt.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/batch_cxr.png')
      
      print(np.max(np.array(batch_y)))
      batch_y = batch_y * 255
      grid_img = make_grid(batch_y, nrow=4)
      plt.imshow(grid_img.permute(1, 2, 0))
      plt.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/batch_masks.png')
      
      optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,  mode='min', factor=0.9, patience=5, threshold=1e-10, 
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08, verbose=True)

      val_loader = create_dataloader(bs=4, custom_dataloader=CustomDataLoader, dataframe=val_data, num_workers=8)
      
      model = train(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer_ft, loss_fn=DiceBCELoss(), k=fold_no, patience=20, num_epochs=200, scheduler=scheduler)
      fold_no = fold_no + 1
