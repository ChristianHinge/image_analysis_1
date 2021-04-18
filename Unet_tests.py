# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:08:36 2021

@author: simon
"""


#%%
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
import sys
import numpy as np
import os


class Unet():
    def __init__(self,X_dim,Y_dim,n_channels):
        # Input shape
        self.img_rows = X_dim
        self.img_cols = Y_dim
        self.channels = n_channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Number of filters in the first layer
        self.gf = 32

        optimizer = Adam(0.0002, 0.5)

        # Build the Unet
        self.Unet_model = self.build_Unet()

        # Input images 
        img_in = Input(shape=self.img_shape)
        
        # Compile model 
        self.Unet_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        
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
import cv2

X_re = np.zeros((5,512,512,2))

for ii,s in enumerate(X):
    X_re[ii,:,:,:] = cv2.resize(s, (512,512), interpolation = cv2.INTER_CUBIC)

#%%

Unet_model = Unet(X_re.shape[1],X_re.shape[2],2)

y = Unet_model(X_re[:,:,:,:],training=False)



















#%%

def Unet(img,gf):
    """U-Net"""
    initializer = tf.initializers.GlorotUniform()
    
    def conv2d(layer_input, n_filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(n_filters, kernel_size=f_size, strides=2, padding='same',kernel_initializer=initializer,use_bias=True)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, n_filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(n_filters, kernel_size=f_size, strides=1, padding='same', activation='relu',kernel_initializer=initializer,use_bias=True)(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img.shape)
    
    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
    
    return Model(d0, output_img)


Unet_model = Unet(X_re[0,:,:,:],32)


Unet_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#infer
y = Unet_model(X_re[:,:,:,:],training=False)



























#%%


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result



def Generator(gf):
  inputs = tf.keras.layers.Input(shape=[512, 512, 2])

  down_stack = [
    downsample(gf, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
    downsample(gf, 4),  # (bs, 64, 64, 128)
    downsample(gf, 4),  # (bs, 32, 32, 256)
    downsample(gf, 4),  # (bs, 16, 16, 512)
    downsample(gf, 4),  # (bs, 8, 8, 512)
    downsample(gf, 4),  # (bs, 4, 4, 512)
    downsample(gf, 4),  # (bs, 2, 2, 512)
    downsample(gf, 4),  # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(gf, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
    upsample(gf, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
    upsample(gf, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
    upsample(gf, 4),  # (bs, 16, 16, 1024)
    upsample(gf, 4),  # (bs, 32, 32, 512)
    upsample(gf, 4),  # (bs, 64, 64, 256)
    upsample(gf, 4),  # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(1, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



generator = Generator(32)

output1 = generator.predict(X_re[:,:,:,:])

output=generator(X_re[:,:,:,:],training=False)


