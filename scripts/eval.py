import torch 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import argparse
import copy
from tqdm import tqdm
import time

from matplotlib import ticker

from unet import VGGNestedUNet, DiceBCELoss

from dataloaders import CustomDataLoader, TestDataLoader, create_dataloader

from utils import dice2D, add_colorbar, dice_coef, bbox, draw_bbox

import diptest
import cv2

plt.style.use(['science','nature'])

# component analysis - open cv https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
def iou_bin_(mask1, mask2):
    print(np.unique(np.array(mask1), return_counts=True))
    print(np.unique(np.array(mask2), return_counts=True))
    intersection = (mask1 * mask2).sum()
    if intersection == 0:
        return 0.0
    union = torch.logical_or(mask1, mask2).to(torch.int).sum()
    return intersection / union
# identify > 2 components
# eliminate component with greatest uncertainty
def component_analysis(x, uncertainty_map):
 #   x = copy.deepcopy(x)
    #uncertainty_map = uncertainty_map.copy()
    x = (x * 255).astype(np.uint8)

    numLabels, labels,stats, centroids = cv2.connectedComponentsWithStats(x, 127., cv2.CV_32S)

    for j in range(1,numLabels - 2):
  #        x = x.copy()
   #       uncertainty_map = uncertainty_map.copy()
          # loop over the number of unique connected component labels
          numLabels, labels,stats, centroids = cv2.connectedComponentsWithStats(x, 127. ,cv2.CV_32S)
  
          componentUnc = []
  
          for i in range(0, numLabels):
              # extract the connected component statistics for the current
              # label
              # if this is the first component then we examine the
              # *background* (typically we would just ignore this
              # component in our loop)
              if i == 0:
                  text = "examining component {}/{} (background)".format(
                      i + 1, numLabels)
              # otherwise, we are examining an actual connected component
              else:
                  text = "examining component {}/{}".format( i + 1, numLabels)
              # print a status message update for the current connected
              # component
              print("[INFO] {}".format(text))
              area = stats[i,cv2.CC_STAT_AREA]
              componentMask = (labels == i).astype("uint8")
  
              componentUnc.append(np.sum(componentMask * uncertainty_map)/area)
          
          most_unc_component = np.argmax(componentUnc)
      
          x[labels==most_unc_component] = 0
          uncertainty_map[labels==most_unc_component] = 0

    return x/255., uncertainty_map


# hartigans dip test
def test_bimodal(x, unc_threshold): # input prediction uncertainty array
    # both the dip statistic and p-value
    x = x.flatten()
    dip, pval = diptest.diptest(x)
    total_uncertainty  = np.sum(x) 

    if dip < 0.05: #0.1?
        bimodal = True
    else:
        bimodal = False
        
#    print(bimodal)

    if bimodal == True and total_uncertainty > unc_threshold:
        reject = True
    else:
        reject = False
    return reject


# HDS values range from 0 to 1 with values less than .05 indicating significant bimodality, 
# and values greater than .05 but less than .10 suggesting bimodality with marginal significance".

def post_processing(x, uncertainty_map, unc_threshold):
#    x = copy.deepcopy(x)
    #uncertainty_map = copy.deepcopy(uncertainty_map)
    print(test_bimodal(uncertainty_map, unc_threshold))

    if test_bimodal(uncertainty_map, unc_threshold)==True:
        x = np.ones((480,480,1))
        skip = True
    
    else:
        x, uncertainty_map = component_analysis(x, uncertainty_map)
        skip = False

        
    return x, uncertainty_map, skip


def unsupervised_eval(model, test_loader, unc_threshold, do_post_processing=True):
    with torch.no_grad():
        i = 0
        for data in tqdm(test_loader):
            i = i + 1
            image, gt = data #img, class
            image = image.cuda()
            print(image.shape)

            if do_post_processing:
                raw_pred, pred, aleatoric, epistemic, entropy, mi, variance = inference(model, image[0], gt[0], supervised=False, N=10)
                processed_pred, processed_uncertainty, skip = post_processing(pred, epistemic, unc_threshold)
                if skip == False:
                  processed_pred = np.squeeze(processed_pred)
                  processed_pred_roi = draw_bbox(bbox(processed_pred))
                                  
                  pred = torch.tensor(raw_pred)
                  pred = torch.where(pred > 0.5, 1, 0)
                  pred = pred.detach().cpu().numpy()
                  pred_roi = draw_bbox(bbox(pred[0]))
                  
                  image = image.detach().cpu().numpy()
                  image = np.squeeze(image[0][0])
                  
                  pred1 = pred * image
                  processed_pred1 = processed_pred * image
                  
                  pred_roi1 = pred_roi * image
                  processed_roi1 = processed_pred_roi * image
                  
                  plt.rcParams['image.cmap'] = 'gray'
                  fig, ax = plt.subplots(2,3,figsize=(20,10))
  
                  total_unc = np.sum(epistemic)
  
                  fig.suptitle('Prediction Uncertainty {:.2}\n'.format(total_unc), y=1.0, fontsize=14)   
  
                  cax0 = ax[0,0].imshow(image)
                  plt.colorbar(cax0, ax=ax[0,0])
                  ax[0,0].set_title('Chest X-ray', fontsize=10)
  
                  cax2 = ax[0,1].imshow(np.squeeze(pred1))
                  plt.colorbar(cax2, ax=ax[0,1])
                  ax[0,1].set_title('Semantic segmentation', fontsize=14)
  
                  cax2 = ax[0,2].imshow(np.squeeze(pred_roi1))
                  plt.colorbar(cax2, ax=ax[0,2])
                  ax[0,2].set_title('Predicted ROI', fontsize=14)        
  
                  cax1 = ax[1,0].imshow(np.squeeze(epistemic))
                  plt.colorbar(cax1, ax=ax[1,0])
                  ax[1,0].set_title('Uncertainty map', fontsize=14)        
  
                  cax1 = ax[1,1].imshow(np.squeeze(processed_pred1))
                  plt.colorbar(cax1, ax=ax[1,1])
                  ax[1,1].set_title('Post-processed prediction', fontsize=14)    
  
                  cax1 = ax[1,2].imshow(np.squeeze(processed_roi1))
                  plt.colorbar(cax1, ax=ax[1,2])
                  ax[1,2].set_title('Post-processed ROI', fontsize=14) 
                  
                  for a in ax.flatten(): a.axis('off')
                  if gt.detach().cpu().numpy() == 1:
                      fig.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/test_prediction_uncertainty_postprocessing_covid{:03d}.png'.format(i), dpi=300)
                  if gt.detach().cpu().numpy() == 0:
                      fig.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/test_prediction_uncertainty_postprocessing_noncovid{:03d}.png'.format(i), dpi=300)
                else:
                  pass



def model_eval(model, test_loader, unc_threshold, do_post_processing=False):
    
    dice = []
    precision = []
    recall = []
    iou = []

    roi_dice = []
    roi_precision = []
    roi_recall = []
    roi_iou = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            image, gt = data

            image = image.cuda()
            gt = gt.cuda()

            if do_post_processing:
                raw_pred, pred, aleatoric, epistemic, entropy, mi, variance, error, scores = inference(model, image[0], gt[0] , N=10)
                raw_pred = raw_pred.detach().cpu()         
  
                gt_roi = torch.tensor(draw_bbox(bbox(gt.detach().cpu().numpy()[0][0])))

                processed_pred, processed_uncertainty, skip = post_processing(pred, epistemic, unc_threshold)
              #  print(np.unique(processed_pred, return_counts=True))             
                pred_roi = torch.tensor(draw_bbox(bbox(processed_pred)))
                pred = torch.tensor(raw_pred)
                gt = gt.detach().cpu()
            else:
                skip = False

                pred = model(image)[-1][0]
                pred = torch.sigmoid(pred)
                pred = torch.where(pred > 0.5, 1, 0)    
                gt = torch.where(gt >= 0.5, 1, 0)     
                #print('bbox g', bbox(gt.detach().cpu().numpy()[0][0]))      
                pred_roi = torch.tensor(draw_bbox(bbox(pred.detach().cpu().numpy()[0])))
                gt_roi = torch.tensor(draw_bbox(bbox(gt.detach().cpu().numpy()[0][0])))
            
            if skip == True:
                pass

            else:
                dice.append(dice_coef(gt, pred, sigmoid=False).detach().cpu().numpy())
                precision.append(prec(gt, pred).detach().cpu().numpy())
                recall.append(rec(gt, pred).detach().cpu().numpy())
                iou.append(iou_(gt, pred).detach().cpu().numpy())
                roi_dice.append(dice_coef(gt_roi, pred_roi, sigmoid=False).detach().cpu().numpy())
                roi_precision.append(prec(gt_roi, pred_roi).detach().cpu().numpy())
                roi_recall.append(rec(gt_roi, pred_roi).detach().cpu().numpy())
                roi_iou.append(iou_(gt_roi, pred_roi))    

    
    return np.mean(np.array(dice)), np.mean(np.array(precision)), np.mean(np.array(recall)), np.mean(np.array(iou)), np.mean(np.array(roi_dice)), \
    np.mean(np.array(roi_precision)), np.mean(np.array(roi_recall)), np.mean(np.array(roi_iou))

        
        


def Entropy(X, axis=-1):
    '''
    Helper function to compute entropy: all uncertainty metrics computed in calc_Uncertainty()
    '''

    return -1* np.sum(X * np.log(X+1e-12), axis=axis)
  
# calculate uncertainty metrics
def calc_Uncertainty(preds):
    # calculate mean
    mean_preds = np.mean(preds, axis=0)
    # calculate entropy
    entropy=Entropy(np.mean(preds, axis=0),axis=-1)
    # Expected entropy of the predictive under the parameter posterior
    entropy_exp = np.mean(Entropy(preds, axis=0))
    # calculate mutual info
    mutual_info = entropy - entropy_exp  # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf
    # calculate variance
    variance = np.std(preds[:], 0)
    # calculate aleatoric uncertainty
    aleatoric = np.mean(preds*(1-preds), axis=0)  # approximate with tta?
    # calculate epistemic uncertainty
    epistemic = np.mean(preds**2, axis=0) - np.mean(preds, axis=0)**2
    # overall alertoric + epistemic
    overall= aleatoric + epistemic
    return mean_preds, entropy,mutual_info, variance, aleatoric, epistemic , overall
    

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
        
def inference(model, image, y_true, supervised=True, N=10):
    
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    y_true = torch.unsqueeze(y_true, 0)
    
    # perform N stochastic forward passes and then append the preds
    preds = []
    for n in range(N):
        pred=model(image)[-1][0]
        pred = torch.sigmoid(pred)
        
        preds.append(pred.detach().cpu().numpy())

    preds = np.array(preds)
    preds_n = torch.tensor(preds).permute(0,2,3,1).cpu().numpy()
    #y_true = torch.tensor(y_true).permute(0,2,3,1).cpu().numpy()
   # image = torch.tensor(image).permute(0,2,3,1).cpu().numpy()

    # calculate the uncertainty metrics
    prediction, entropy, mutual_info, variance, aleatoric, epistemic, overall = calc_Uncertainty(preds_n)
    
    pred=model(image)[-1][0]
    # calculate the accuracy metrics
    prediction = np.where(preds_n[0] > 0.5, 1, 0)

    if supervised:
        dice = dice_coef(y_true, pred, sigmoid=True)
        precision = prec(y_true, pred)
        recall = rec(y_true, pred)
        iou = iou_(y_true, pred)
        iou_loss = iou_loss_(y_true, pred)
        
        y_true = torch.tensor(y_true).permute(0,2,3,1).cpu().numpy()
        error=y_true[0]-prediction
        prediction = np.where(preds_n[0] > 0.5, 1, 0)
        return torch.sigmoid(pred), np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic),np.squeeze(entropy), np.squeeze(mutual_info),np.squeeze(variance), \
         np.squeeze(error), (dice, precision, recall)
    
    else:
        return torch.sigmoid(pred), np.squeeze(prediction), np.squeeze(aleatoric), np.squeeze(epistemic),np.squeeze(entropy), np.squeeze(mutual_info),np.squeeze(variance)
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--data_csv', default="/MULTIX/DATA/HOME/COVID_QU_Ex/covid_qu_ex.csv", type=str, help='Path to data file')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--n_test', default=20)
    parser.add_argument('--test', default=None, type=str, help='Choose on of: [ltht, nccid_test, nccid_val, nccid_leeds, chexpert, custom]')
    parser.add_argument('--uncertainty_threshold', type=int)
    args = parser.parse_args()

    model = VGGNestedUNet(num_classes=1, deep_supervision=True).cuda()
    
    model.load_state_dict(torch.load("/MULTIX/DATA/HOME/mcd_vgg_nested_unet_cxr_1.pth"))

    full_data = pd.read_csv(args.data_csv)
    seed = 0
    np.random.seed(seed)
    
    if args.test == None:
        test_data = full_data[full_data['split']=='Test']
        test_data=test_data.reset_index(drop=True)
        test_loader = create_dataloader(bs=50, custom_dataloader=CustomDataLoader, dataframe=test_data, train=False, num_workers=0)
    
        x, y = next(iter(test_loader))
        
        n = np.random.randint(0, x.shape[0])
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(x[n][0])
        ax[1].imshow(np.squeeze(y[n][0]))
        for a in ax: a.axis('off')
        plt.savefig("/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/test_input_example.png")
        # save
        
        x = x.cuda()
        y = y.cuda()
    
        X_predict, Y_predict = x, y
        num = 20
        plt.rcParams['image.cmap'] = 'gray'
        import matplotlib.gridspec as gridspec

        for i in range(num):
            sample = np.random.randint(0,len(X_predict))
            image = X_predict[sample]
            true  = Y_predict[sample]
            
            raw_pred, prediction, aleatoric, epistemic, entropy, mi, variance, error, scores = inference(model, image, true , N=10)

            pred_bbox = draw_bbox(bbox(prediction))

            gt_bbox = draw_bbox(bbox(true.detach().cpu().numpy()[0]))

            processed_prediction, processed_uncertainty, skip = post_processing(prediction, epistemic, 500)
            processed_roi = draw_bbox(bbox(processed_prediction))

            true = np.squeeze(true)
                
            n = np.random.randint(0,num)
            fig, ax = plt.subplots(3,3,figsize=(20,10))

            total_unc = np.sum(epistemic)

            dice, precision, recall = scores
            fig.suptitle('Dice: {:.2f} | Uncertainty {:.2}\n'.format(dice, total_unc), y=1.0, fontsize=14)   

            image = image.detach().cpu().numpy()
            cax0 = ax[0,0].imshow(image[0])
            plt.colorbar(cax0, ax=ax[0,0])
            ax[0,0].set_title('Chest X-ray')

            cax2 = ax[0,1].imshow(true.detach().cpu().numpy())
            plt.colorbar(cax2, ax=ax[0,1])
            ax[0,1].set_title('Ground truth segmentation')

            cax2 = ax[0,2].imshow(gt_bbox)
            plt.colorbar(cax2, ax=ax[0,2])
            ax[0,2].set_title('Ground truth ROI')        
        
            cax1 = ax[1,0].imshow(prediction)
            plt.colorbar(cax1, ax=ax[1,0])
            ax[1,0].set_title('Segmentation prediction')

            cax1 = ax[1,1].imshow(pred_bbox)
            plt.colorbar(cax1, ax=ax[1,1])
            ax[1,1].set_title('Prediction ROI')     

            cax1 = ax[1,2].imshow(epistemic)
            plt.colorbar(cax1, ax=ax[1,2])
            ax[1,2].set_title('Uncertainty')        

            cax1 = ax[2,0].imshow(processed_prediction)
            plt.colorbar(cax1, ax=ax[2,0])
            ax[2,0].set_title('Post-processed prediction')    

            cax1 = ax[2,1].imshow(processed_roi)
            plt.colorbar(cax1, ax=ax[2,1])
            ax[2,1].set_title('Post-processed ROI')   

            cax1 = ax[2,2].imshow(processed_uncertainty)
            plt.colorbar(cax1, ax=ax[2,2])
            ax[2,2].set_title('Post-processed Uncertainty')            

            for a in ax.flatten(): a.axis('off')
            fig.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/prediction_uncertainty_postprocessing_{:03d}.png'.format(i), dpi=300)

        
        for i in range(num):
            sample = np.random.randint(0,len(X_predict))
            image = X_predict[sample]
            true  = Y_predict[sample]
            
            raw_pred, prediction, aleatoric, epistemic, entropy, mi, variance, error, scores = inference(model, image, true , N=10)
            true = np.squeeze(true)
                
            n = np.random.randint(0,num)
            fig, ax = plt.subplots(3,3,figsize=(20,10))
            
            dice, precision, recall = scores
            fig.suptitle('Dice: {:.2f}\n'.format(dice), y=1.0, fontsize=14)
            
            image = image.detach().cpu().numpy()
            cax0 = ax[0,0].imshow(image[0])
            plt.colorbar(cax0, ax=ax[0,0])
            ax[0,0].set_title('Chest X-ray')
        
            cax1 = ax[0,1].imshow(prediction)
            plt.colorbar(cax1, ax=ax[0,1])
            ax[0,1].set_title('Segmentation prediction')
        
            cax2 = ax[0,2].imshow(true.detach().cpu().numpy())
            plt.colorbar(cax2, ax=ax[0,2])
            ax[0,2].set_title('Ground truth segmentation')
            
            cax3 = ax[1,0].imshow(aleatoric)
            plt.colorbar(cax3, ax=ax[1,0])
            ax[1,0].set_title('Aleatoric uncertainty')
            
            cax4 = ax[1,1].imshow(epistemic)
            plt.colorbar(cax4, ax=ax[1,1])
            ax[1,1].set_title('Epistemic uncertainty')
        
            cax5 = ax[1,2].imshow(aleatoric+epistemic)
            plt.colorbar(cax4, ax=ax[1,2])
            ax[1,2].set_title('Uncertainty (combined)')
        
            
            cax6 = ax[2,0].imshow(entropy)
            plt.colorbar(cax3, ax=ax[2,0])
            ax[2,0].set_title('Entropy')
            
            cax7 = ax[2,1].imshow(mi)
            plt.colorbar(cax4, ax=ax[2,1])
            ax[2,1].set_title('Mutual Information')
        
            cax8 = ax[2,2].imshow(error)
            plt.colorbar(cax4, ax=ax[2,2])
            ax[2,2].set_title('Error')

            for a in ax.flatten(): a.axis('off')
                
            fig.savefig('/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/prediction_uncertainty_{:03d}.png'.format(i), dpi=300)
            plt.show()
            plt.close()


        test_data = test_data.sample(frac=1, random_state=22)
        test_data = test_data.reset_index(drop=True)
        test_data = test_data[:2000]
        test_loader = create_dataloader(bs=1, custom_dataloader=CustomDataLoader, dataframe=test_data, train=False, num_workers=4)
        dice, precision, recall, iou, roi_dice, roi_precision, roi_recall, roi_iou = model_eval(model, test_loader, args.uncertainty_threshold, do_post_processing=False)
        print(f"dice: {dice} | precision: {precision} | recall: {recall}, iou: {iou} | \n dice (roi): {roi_dice} | precision (roi): {roi_precision} | recall (roi): {roi_recall} | iou (roi): {roi_iou}")
        
        dice1, precision1, recall1, iou1, roi_dice1, roi_precision1, roi_recall1, roi_iou1 = model_eval(model, test_loader, args.uncertainty_threshold, do_post_processing=True)
        print('post processing')
        print(f"dice: {dice1} | precision: {precision1} | recall: {recall1}, iou: {iou1} | \n dice (roi): {roi_dice1} | precision (roi): {roi_precision1} | recall (roi): {roi_recall1} | iou (roi): {roi_iou1}")
        results =  {'dice':[dice, dice1], 'precision':[precision,precision1], 'recall':[recall, recall1], 'iou':[iou, iou1], 'dice (roi)':[roi_dice, roi_dice1],\
         'precision (roi)':[roi_precision, roi_precision1], 'recall (roi)':[roi_recall, roi_recall1], 'iou (roi)':[roi_iou, roi_iou1]} 
        df = pd.DataFrame.from_dict(results)
        df.to_csv(f'/MULTIX/DATA/HOME/covid-19-benchmarking/uncertainty_unet/results{args.uncertainty_threshold}.csv')
 
    else:
        test_data = full_data[100:200]
        test_data = test_data.reset_index(drop=True)
        test_loader = create_dataloader(bs=1, custom_dataloader=TestDataLoader, dataframe=test_data, train=False, num_workers=4)
        unsupervised_eval(model, test_loader, unc_threshold=800)
        

