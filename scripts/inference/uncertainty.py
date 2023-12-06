import numpy as np
import diptest
import cv2

# calculate uncertainty metrics
def calc_uncertainty(preds):
    # calculate mean
    mean_preds = np.mean(preds, axis=0)
    # calculate entropy
    entropy=entropy(np.mean(preds, axis=0),axis=-1)
    # Expected entropy of the predictive under the parameter posterior
    entropy_exp = np.mean(entropy(preds, axis=0))
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


def component_analysis(x, uncertainty_map):
    x = (x * 255).astype(np.uint8)

    numLabels, labels,stats, centroids = cv2.connectedComponentsWithStats(x, 127., cv2.CV_32S)

    for j in range(1,numLabels - 2):
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
    x = x.flatten()
    dip, pval = diptest.diptest(x)
    total_uncertainty  = np.sum(x) 

    if dip < 0.05:
        bimodal = True
    else:
        bimodal = False
        
    if bimodal == True and total_uncertainty > unc_threshold:
        reject = True
    else:
        reject = False
    return reject