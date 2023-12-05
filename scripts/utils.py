
import torch
import matplotlib.pyplot as plt
import numpy as np
import itertools
from functools import reduce
import cv2
import copy
from mpl_toolkits import axes_grid1

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def dice2D(a,b):
    #https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
    #https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    intersection = np.sum(a[b==1])
    dice = (2*intersection)/(np.sum(a)+np.sum(b))
    if (np.sum(a)+np.sum(b))==0: #black/empty masks
        dice=1.0
    return(dice)

def mask_from_img(img):
    """create mask from prediction
    args:
        img: mask prediction - open-cv numpy array image"""
    return (img >= 0.5).astype(np.float32)

def mask_from_bbox(img, bbox):
    """segment image according to bbox coordinates
    args:
        img: image as open-cv numpy array
        bbox: numpy array - [xmin, xmax, ymin, ymax] coordinates"""
#    img = copy.deepcopy(img)
    bbox_mask = np.zeros(img.shape,np.float32)
    xmin, xmax, ymin, ymax = bbox
    bbox_mask[ymin:ymax,xmin:xmax] = img[ymin:ymax,xmin:xmax]
    bbox_mask = bbox_mask.astype(np.float32)
    return bbox_mask
  
def bbox(img, input='prediction'):
    """extract bounding box coords from input
    args:
        img: image as open-cv numpy array
         input=mask: str arg - mask or prediction"""
#    print(np.shape(img))
    if input == 'mask':
        a = np.where(np.array(img) == np.max(np.array(img)))

    elif input == 'prediction':
        a = np.where(np.array(img) >= 0.5)
 #   print('a0',a[0],np.shape(a))
    bbox = np.min(a[1]), np.max(a[1]), np.min(a[0]), np.max(a[0]) # extract coords - xmin, xmax, ymin, ymax
    return bbox

def draw_bbox(bbox_coords):
    print(bbox_coords)
    bbox_mask = np.zeros((480,480), np.float32)
    xmin, xmax, ymin, ymax = bbox_coords
    bbox_mask[ymin:ymax,xmin:xmax] = 1
    return bbox_mask

def visualize_bbox(img, bbox, color=(201, 58, 64), thickness=5):  #https://www.kaggle.com/blondinka/how-to-do-augmentations-for-instance-segmentation
    """ add bboxes to images 
    args:
        img : image as open-cv numpy array
        bbox : boxes as a list or numpy array in pascal_voc format [x_min, y_min, x_max, y_max]  
        color=(255, 255, 0): boxes color 
        thickness=2 : boxes line thickness
    """

    # draw bbox on img
 #   img = copy.deepcopy(img)
    xmin, xmax, ymin, ymax = bbox
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)

    img = img.astype(np.float32)
    return img

def segment_mask(img, mask):
    """ extract region from predicted bbox
    args:
        img: image as open-cv numpy array
        mask: uint8 type numpy array
    """
  #  img = copy.deepcopy(img)
    mask = mask.transpose((1,2,0))
    img = img * mask

    return img.astype(np.float32)

def reverse_transform(inp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    inp = inp.numpy().transpose((1, 2, 0))
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp).astype(np.float32)/np.max(inp)
    return inp

def plot_img_array(img_array, idx, model_num, ncol=5, img_class=None):
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))
    print(len(img_array))
    print(img_class)
    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])
    
    if img_class != None:
        for j in range(4):
            plots[j, 0].set_ylabel(img_class[j])

    plt.savefig(f"/MULTIX/DATA/HOME/lung_segmentation/segmentation_results/seg_data_vggnestedunet_{model_num}_{idx}")

def plot_side_by_side(img_arrays, idx, model_num, img_class=None):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))
    print(np.array(flatten_list).shape)

    plot_img_array(np.array(flatten_list),idx, model_num, ncol=len(img_arrays), img_class=img_class)

def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape


    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.float32)/np.max(colorimg)

############################# Define Loss & Metrics ###########################

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
