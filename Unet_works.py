#%%
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os 

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

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true + y_pred)

  return 1 - numerator / denominator

  
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
              loss=tf.keras.losses.BinaryCrossentropy (),
              metrics=tf.keras.metrics.BinaryCrossentropy ())
        
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

import wandb
from wandb.keras import WandbCallback
from keras.callbacks import Callback

def log_image(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = Unet_model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)
    plt.figure(figsize=(10,3))
    plt.subplot(1,4,1)
    plt.imshow(X_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(1,4,2)
    plt.imshow(X_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(1,4,3)
    plt.imshow(test_pred_raw[0,:,:,0])
    plt.axis("off")
    plt.title("Predicted")
    plt.subplot(1,4,4)
    plt.imshow(Y_test[0,])
    plt.axis("off")
    plt.title("True")
    plt.tight_layout()
    # Log the confusion matrix as an image summary.

    wandb.log({"test:" : plt}, step=epoch)
    #

"""
# Define the per-epoch callback.
image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)

with open("wandb.key" , "r") as handle:
    wandb_key = handle.readlines()[0]

wandb.login(key=wandb_key)
wandb.init(project='unet', entity='keras_krigere')

#batch_size, X_dim, Y_dim, n_channels = X_re.shape

dl = DataLoader(IDs[:2],batch_size=2)

X_test_, Y_test_ = dl[0]

X_test = X_test_[np.newaxis,0,]
Y_test = Y_test_[np.newaxis,0,]

unet = Unet(512,512,2,32)
Unet_model = unet.Unet_model



# Define the per-epoch callback.
image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)

#log_confusion_matrix(1,1)

#%%

Unet_model.fit(dl,epochs=10,
    callbacks=[image_callback,WandbCallback()])

Unet_model.save('path/to/location')

#%%
1+1
"""