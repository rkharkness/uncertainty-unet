
import torch

import gc

from torch.utils.data import Dataset 
import numpy as np

import pandas as pd 
import matplotlib.pyplot as plt

import argparse 
import tempfile 
import datetime

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
#from memory_profiler import profile
from pydicom.uid import ExplicitVRLittleEndian

from pydicom.uid import UID
from tqdm import tqdm
import time
from unet import VGGNestedUNet

from eval import inference, post_processing
from utils import draw_bbox, bbox

import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import os

#import SimpleITK as sitk


class DataLoader(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        img_path = self.df['cxr_path'][index]
        label = self.df['xray_status'][index]

        return img_path, label

    def __len__(self):
        return len(self.df)

#/MULTIX/DATA/INPUT_NCCID/nccid_dcm/Covid1_1.2.826.0.1.3680043.9.3218.1.1.1575273888.
#\1542.1586418238880.4.0_1.2.826.0.1.3680043.9.3218.1.1.1575273888.1542.1586418238880.5.0_1.2.826.0.1.3680043.9.3218.1.1.1575273888.1542.1586418238880.7.0.dcm
def process_path(path, test):
    if test == 'nccid':
        root = '/MULTIX/DATA/nccid_dcm_seg'
        patientid = path.split('/')[5:]
        patientid = '/'.join(patientid)
        new_path = os.path.join(root, patientid)
        print(new_path)
    # /MULTIX/DATA/INPUT/ltht_dcm14_21/117761D87432D90273967A49142EA3218F114781A3D3DD16E19EC231AC81586F_XR Chest 08111895_AP_IM1.DCM
    if test == 'ltht_binary14_21':
        root = '/MULTIX/DATA/INPUT/ltht_dcm_seg'
        patientid = path.split('/')[4:]
        patientid = '_'.join(patientid)
        new_path = os.path.join(root, patientid)

    # /MULTIX/DATA/HOME/Infection Segmentation Data/Infection Segmentation Data/Test/COVID-19/images/covid_1579.png
    if test == 'covidgr':
        root = '/MULTIX/DATA/HOME/covid-19-benchmarking/data/covidgr_seg'
        patientid = path.split('/')[4:]
        patientid = '_'.join(patientid)
        new_path = os.path.join(root, patientid)           
    
    return new_path
    

def seg_checker(path, test):
    print('checking segmentation ...')
    if test == "nccid":
        ds = pydicom.dcmread(path)
        arr  = ds.pixel_array
#       
    arr = arr/np.max(arr.flatten())

    plt.figure()
    plt.imshow(arr)
    plt.savefig(f"/MULTIX/DATA/scripts/uncertainty_unet/seg_check_{test}.png",cmap='grey')
    plt.figure()
    plt.hist(arr.flatten())
    plt.savefig(f"/MULTIX/DATA/scripts/uncertainty_unet/seg_check_hist_{test}.png")

def prediction_checker(img, idx):
    plt.figure()
    plt.imshow(img.transpose(1,2,0))
    plt.savefig(f"/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/pred_check{idx}.png")

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
#            try:
                image_path, label = data
#
 #               ds = pydicom.dcmread(image_path[0])
                arr = cv2.imread(image_path[0],0)
#                arr  = ds.pixel_array
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
#                    processed_pred_roi1.view('uint16')
                processed_pred_roi1 = np.expand_dims(processed_pred_roi1,-1).astype(np.uint8)
#                    plt.hist(processed_pred_roi1.flatten())
 #                   plt.savefig('/MULTIX/DATA/scripts/uncertainty_unet/pred_hist.png')          
#                prediction_checker(processed_pred_roi1, counter)
                cv2.imwrite(processed_path, processed_pred_roi1)
########                 
#                ds.PixelData = processed_pred_roi1.tobytes()
#                print(processed_pred_roi1.shape)
#                ds.Rows, ds.Columns = processed_pred_roi1.shape[1], processed_pred_roi1.shape[2]
#                ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

#                ds.BitsAllocated = 16
#                ds.BitsStored = 16
#                ds.HighBit = 15
#                ds.save_as(processed_path, write_like_original=False)
#########
                segmentation_path.append(processed_path)
                original_path.append(image_path[0])
                labels.append(label.detach().cpu().numpy()[0][0])
#            except:
#                 print('saving failed')
 #               
                #    if counter == 5:
#                    seg_checker(processed_path, test)

      # #         else:
         #           skip_count = skip_count + 1
       #             pass
      
                #counter = counter + 1
 ##           except:
   #             print('exception')
   #         del image, processed_pred, processed_pred_roi, processed_pred_roi1
    #        gc.collect()
    print('number of skips: ', skip_count)
    return segmentation_path, original_path, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default="/MULTIX/DATA/HOME/covid-19-benchmarking/data/covidgr_data.csv", type=str, help='Path to data file')
    parser.add_argument('--test', default="covidgr", type=str, help='Path to data file')

    args = parser.parse_args()

    model = VGGNestedUNet(num_classes=1, deep_supervision=True).cuda()
    model.load_state_dict(torch.load("/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/mcd_vgg_nested_unet_cxr_1.pth"))

    full_data = pd.read_csv(args.data_csv)
    full_data['cxr_path'] = full_data['paths']
#    full_data['xray_status'] = full_data['FinalPCR']
#    full_data = full_data.drop(index=[1399, 1957, 1960, 1961, 1962, 1959, 1958, 1963, 1964, 1965, 3385, 3376, 3377, 3379, 3380, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3393, 3394, 3395, 3398]).reset_index(drop=True)
#    data_split1 = full_data
# 1678
#1126,  1562, 1675,1675, 1677, 1677, 1678, 1678, 1677,1680,  1682
#    full_data = full_data.drop(index=[3375,3403,3861,3862,4130,4567,4681,4682,4683,4684,4685,4688,4689,4690,4691,4692,4693,4694,4695,4696,4697,4698,4699,4700,4704,4707])
   # full_data = full_data[1399:2000].reset_index(drop=True)
#    full_data = full_data.drop(index=[130,567, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697,698,699,700, 704, 707, 708])
 #   full_data = full_data.reset_index(drop=True)
#    full_data = full_data[:19702]
    full_data = full_data.reset_index(drop=True)
    seed = 0
    np.random.seed(seed)
    bs = 1
    shuffle = False
    num_workers = 1
    pin_memory = False
    
    data = DataLoader(full_data)
    test_loader = torch.utils.data.DataLoader(data, bs,
	                                          shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=pin_memory
	                                          )
    
    segmentation_path, original_path, labels = create_segmentations(model, test_loader, args.test)
    
    segmentation_df = pd.DataFrame()
    
    segmentation_df['cxr_path'] = segmentation_path
    segmentation_df['original_path'] = original_path
    segmentation_df['label'] = labels
    
    segmentation_df.to_csv(f'/MULTIX/DATA/HOME/covid-19-benchmarking/data/full_{args.test}_segmentation_data.csv')
