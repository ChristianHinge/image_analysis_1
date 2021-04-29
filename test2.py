
#%%
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import scipy
from scipy import ndimage as nd



#%% Augmentation

# function cv2_clipped_zoom taken from:
# https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions/37121993#37121993
def cv2_clipped_scale(img, scale_x, scale_y):
    """
    Center scale in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    INPUT
        img : Image array
        scale_x : amount of scale on x axis as a ratio (0 to Inf)
        scale_y : amount of scale on y axis as a ratio (0 to Inf)
    """
    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * scale_y), int(width * scale_x)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / [scale_y,scale_x,scale_y,scale_x]).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def AUG(X,Y,angles = [-15,15],scales =[0.9,1.1]):

    # function can perform horizontal flip, rotation, scaling and normalization
    #
    # INPUT
    # X : [xdim, ydim, 2] image values for brain and bone window
    # Y : [xdim, ydim] segmentation mask image values
    # angles : [angle_low, angle_high] defines range of angles to randomly choose from
    # scales : [scale_low, scale_high] defines range of scaling to randomly choose from
    #
    # OUTPUT
    # X_AUG : [xdim, ydim, 2] image values for augmented brain and bone window
    # Y : [xdim, ydim] augmented segmentation mask image values

    X_AUG = np.copy(X)
    Y_AUG = np.copy(Y)

    # flip (horizontal - xdim)
    flip = np.random.randint(2,size=1) # randomly select 0 or 1 (no flip or flip)

    if flip == 1:
        X_AUG = np.flip(X_AUG,axis=1) # flips bone and brain image
        Y_AUG = np.flip(Y_AUG,axis=1) # flips mask

    # rotate
    # rotation in x,y-plane (axes 0,1)
    # randomize angle selection
    angle = np.random.uniform(low = angles[0], high = angles[1], size = [1,1])
    angle = angle.astype(np.float)
    rows,cols = Y_AUG.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle[0,0],1)
    Y_AUG = cv2.warpAffine(Y_AUG,M,(cols,rows)) # rotate mask
    X_AUG[:,:,0] = cv2.warpAffine(X_AUG[:,:,0],M,(cols,rows)) # rotate bone
    X_AUG[:,:,1] = cv2.warpAffine(X_AUG[:,:,1],M,(cols,rows)) # rotate brain

    # scale
    scale = np.random.uniform(low = scales[0], high = scales[1], size = [1,2])
    scale = scale.astype(np.float)
    # cv2_clipped_scale(image, scale_x, scale_y)
    X_AUG[:,:,0] = cv2_clipped_scale(X_AUG[:,:,0], scale[0,0], scale[0,1])
    X_AUG[:,:,1] = cv2_clipped_scale(X_AUG[:,:,1], scale[0,0], scale[0,1])
    Y_AUG = cv2_clipped_scale(Y_AUG, scale[0,0], scale[0,1])
    
    # elastic transform
    # albumentations.augmentations.geometric.transforms.ElasticTransform (alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, approximate=False, p=0.5)

    return X_AUG, Y_AUG

