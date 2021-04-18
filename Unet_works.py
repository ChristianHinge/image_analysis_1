

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os 

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
from dataloader_test import DataLoader, IDs
import sys
import numpy as np
import os


class Unet():
    def __init__(self,X_dim,Y_dim,n_channels,gf):
        # Input shape
        self.img_rows = X_dim
        self.img_cols = Y_dim
        self.channels = n_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Number of filters in the first layer
        self.gf = gf

        optimizer = Adam(0.0002, 0.5)

        # Build the Unet
        self.Unet_model = self.build_Unet()

        # Compile model 
        self.Unet_model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=tf.keras.metrics.MeanSquaredError())
        
      #  return self.Unet_model


    def build_Unet(self):
        """U-Net"""
        initializer = tf.initializers.GlorotUniform()
        
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same',kernel_initializer=initializer,use_bias=True)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu',kernel_initializer=initializer,use_bias=True)(u)
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
        output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)


#%%
import cv2

X_re = np.zeros((5,512,512,2))
Y_re = np.zeros((5,512,512))

for ii,s in enumerate(X):
    X_re[ii,:,:,:] = cv2.resize(s, (512,512), interpolation = cv2.INTER_CUBIC)

for ii,s in enumerate(Y):
    Y_re[ii,:,:] = cv2.resize(s, (512,512), interpolation = cv2.INTER_CUBIC)



#%%
#batch_size, X_dim, Y_dim, n_channels = X_re.shape

dl = DataLoder(IDs)

unet = Unet(512,512,2,32)
Unet_model = unet.Unet_model
#Unet_model.fit(dl)

#Unet_model.compile(optimizer='adam',
#              loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.metrics.MeanSquaredError())

#y = Unet_model(X_re[:,:,:,:],training = False)




