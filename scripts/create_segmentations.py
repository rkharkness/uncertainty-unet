import torch
from torch.utils.data import Dataset 
import numpy as np

import pandas as pd 

import argparse 
from tqdm import tqdm
import time
from unet import VGGNestedUNet

from data.dataloaders import SegDataLoader
from data.data_process import process_path

from eval import inference, post_processing
from utils import draw_bbox, bbox

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import os

def create_segmentations(model, test_loader, test):
    transforms = A.Compose([
                        ToTensorV2()
                        ])

    segmentation_path = []
    original_path = []
    labels = []
    
    skip_count = 0
    
    counter = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
                counter +=1           
                image_path, label = data

                arr = cv2.imread(image_path[0],0)
                arr = cv2.resize(arr, (480, 480), interpolation = cv2.INTER_AREA)             
                arr = arr/np.max(arr)
                arr = arr.astype(np.float32)
                arr = np.dstack([arr,arr,arr])

                image = transforms(image=arr)["image"]

                label = torch.tensor(label, dtype=torch.float)
                label = torch.unsqueeze(label, 0)
            
                image = image.cuda()
                label = label.cuda()

                raw_pred, pred, aleatoric, epistemic, entropy, mi, variance = inference(model, image, label, supervised=False, N=10)
                processed_pred, processed_uncertainty, skip = post_processing(pred, epistemic, 800)

                processed_pred = np.squeeze(processed_pred)
                processed_pred_roi = draw_bbox(bbox(processed_pred))
                image = image.detach().cpu().numpy()
                image = np.squeeze(image[0])

                processed_pred_roi1 = processed_pred_roi * image
                processed_pred_roi1 = np.round(processed_pred_roi1 * 255.) #65535)

                processed_path = process_path(image_path[0], test)
                processed_pred_roi1 = np.expand_dims(processed_pred_roi1,-1).astype(np.uint8)
                cv2.imwrite(processed_path, processed_pred_roi1)

                segmentation_path.append(processed_path)
                original_path.append(image_path[0])
                labels.append(label.detach().cpu().numpy()[0][0])

    print('number of skips: ', skip_count)
    return segmentation_path, original_path, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default="/MULTIX/DATA/HOME/covid-19-benchmarking/data/covidgr_data.csv", type=str, help='Path to data file')
    parser.add_argument('--model_weights', default="/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/mcd_vgg_nested_unet_cxr_1.pth", type=str, help='Path to weights file')
    parser.add_argument('--test', default="covidgr", type=str, help='Path to data file')

    args = parser.parse_args()

    # define unet++ model
    model = VGGNestedUNet(num_classes=1, deep_supervision=True).cuda()
    model.load_state_dict(torch.load(args.model_weights))

    full_data = pd.read_csv(args.data_csv)
    full_data['cxr_path'] = full_data['paths']
    full_data = full_data.reset_index(drop=True)

    seed = 0
    np.random.seed(seed)

    bs = 1
    shuffle = False
    num_workers = 1
    pin_memory = False
    
    # create segmentation dataloader
    data = SegDataLoader(full_data)
    test_loader = torch.utils.data.DataLoader(data,
                                              bs,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory
	                                          )
    
    segmentation_path, original_path, labels = create_segmentations(model, test_loader, args.test)
    
    # constuct segmentation dataframe
    segmentation_df = pd.DataFrame()
    segmentation_df['cxr_path'] = segmentation_path
    segmentation_df['original_path'] = original_path
    segmentation_df['label'] = labels
    segmentation_df.to_csv(f'/MULTIX/DATA/HOME/covid-19-benchmarking/data/full_{args.test}_segmentation_data.csv')
