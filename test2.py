
#%%
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import scipy
from scipy import ndimage as nd

data_dir = "data/Patients_CT"

#%%
#Slices for patient 67 for whom a segmentation image exists
im_ids = [17,18,19,20,21]

#Dataset dimensions
X_dim = 650
Y_dim = 650
n_images = len(im_ids)
n_channels = 2 


#X: Images, Y: Masks

#Dimensions in the datastructure
#   Dimension 0: The different patients
#   Dimension 1+2: 2D Image data
#   Dimension 3: Channels (e.g. RGB channels or in our case, using both the "Bone" og "Brain" CT images)

X = np.zeros((n_images,X_dim,Y_dim,n_channels))
Y = np.zeros((n_images,X_dim,Y_dim))

#Load dataset 
for i, ID in enumerate(im_ids):

    # Load bone and brain slice
    im_bone  = Image.open(data_dir + "/067/bone/{}.jpg".format(ID))
    im_brain = Image.open(data_dir + "/067/brain/{}.jpg".format(ID))

    # Load segmentation mask
    d = data_dir + "/067/brain/{}_HGE_Seg.jpg".format(ID)
    if os.path.exists(d):
        mask = np.array(Image.open(d))
    else:
        mask = np.zeros((X_dim,Y_dim))

    # Save the images
    X[i,:,:,0] = np.array(im_bone)
    X[i,:,:,1] = np.array(im_brain)

    Y[i,:,:] = mask


#Show example
plt.figure()
plt.imshow(X[0,:,:,0])

plt.figure()
plt.imshow(Y[0,:,:])

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


def AUG(X,Y,angles,scales):
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
    flip = 1
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



#%% PERFORM AUGMENTATION

# define augmentation variables
angles = [-30, 30]      # range of angles as [angle_low, angle_high]
scales = [0.8, 1.2]     # range of scaling as [scale_low, scale_high]

# patient has been selected
for i in range(n_slices)
    # select image slice
    X_slice = X[i,:,:,:]
    Y_slice = Y[i,:,:]

    X_AUG, Y_AUG = AUG(X_slice,Y_slice,angles,scales)



#%% U-net 

#from __future__ import print_function, division
import scipy

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import sys
import numpy as np
import os


class Unet():
    
    def __init__(self,batch_size,X_dim,Y_dim,n_channels):
        # Input shape
        self.batch_size = batch_size
        self.img_rows = X_dim
        self.img_cols = Y_dim
        self.channels = n_channels
        self.img_shape = (self.batch_size, self.img_rows, self.img_cols, self.channels)

        # Number of filters in the first layer
        self.gf = 64

        optimizer = Adam(0.0002, 0.5)

        # Build the Unet
        self.Unet_model = self.build_Unet()

        # Input images 
        img_in = Input(shape=self.img_shape)

        img_out = self.Unet_model(img)
        return img_out


    def build_Unet(self):
        """U-Net"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

#%%

batch_size, X_dim, Y_dim, n_channels = X.shape

Unet_model = Unet(X_dim,Y_dim,batch_size,n_channels)

X_forwardpass = Unet_model(X) 
