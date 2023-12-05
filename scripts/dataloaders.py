from utils import bbox
from utils import draw_bbox

from  torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2
import torch
import pydicom

import numpy as np

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataloader import default_collate
############################# Dataloaders ###########################


class CustomDataLoader(Dataset):
    def __init__(self, df, transforms=None):
        self.transforms = transforms
        self.df = df

    def __getitem__(self, index):
        mask_path = self.df['mask'][index]
        cxr_path = self.df['img_files'][index]

        assert cxr_path.split('/')[10] == mask_path.split('/')[10]

        image = cv2.imread(cxr_path,1)
        image = cv2.resize(image, (480,480))

        masks = cv2.imread(mask_path,0)
        masks = cv2.resize(masks, (480,480))

        masks = [masks]
        
        if self.transforms is not None:
            augmented = self.transforms(image=image, masks=masks)
            image = augmented['image']
            masks = augmented['masks']

        return torch.as_tensor(image, dtype=torch.float32)/255.0, torch.as_tensor(np.array(masks), dtype=torch.float32)/255.0
    
    def __len__(self):
        return len(self.df)

class TestDataLoader(Dataset):
    def __init__(self, df, transforms):
        self.df = df

        self.transforms = A.Compose([
                             ToTensorV2()
                             ])

    def __getitem__(self, index):
        path = self.df['cxr_path'][index]
        label = self.df['xray_status'][index]
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array

        arr = cv2.resize(arr, (480, 480), interpolation = cv2.INTER_AREA)
        
        arr = arr/np.max(arr)
        arr = arr.astype(np.float32)
        
        arr = np.dstack([arr,arr,arr])

        image = self.transforms(image=arr)["image"]

        label = torch.tensor(label, dtype=torch.float)
        label = torch.unsqueeze(label, 0)
                
        return image, label

    def __len__(self):
        return len(self.df)

def create_dataloader(bs, dataframe, custom_dataloader, num_workers, train=False, shuffle=False, pin_memory=False):

    if train == False:
        shuffle = True
        transform = A.Compose([
                            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.2),
                            A.HorizontalFlip(p=0.2),
                            A.OneOf([
                                A.RandomBrightnessContrast(p=0.2),  
                                A.RandomContrast(p=0.2)
                                ], p=0.2),
                            A.RandomBrightnessContrast(p=0.2),  
                            A.RandomContrast(p=0.2),  
                            A.ColorJitter(0.2),
                            ToTensorV2()])

    else:
        shuffle = False
        transform = A.Compose([
                             ToTensorV2()
                             ])

    data = custom_dataloader(dataframe, transforms=transform)
    dataloader = torch.utils.data.DataLoader(data, bs,
	                                        shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory
	                                          )
    return dataloader