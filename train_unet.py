# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:14:51 2021

@author: simon
"""
#%%
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from Unet_works import Unet
from dataloader_test import DataLoader, IDs, preprocess
from data_split import get_data_split_IDs
from test2 import AUG
from tensorflow import keras
from keras.callbacks import CSVLogger

#from Unet_works import log_image
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import Callback
import numpy as np

import tensorflow as tf
assert tf.test.is_gpu_available()
assert tf.test.is_built_with_cuda()
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt

print(device_lib.list_local_devices())

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#%%

data_dir = "data/Patients_CT"
norm_dir = "/data/normalized"
checkpoint_dir = "checkpoints/model_{epoch:02d}.hdf5"


cwd = os.getcwd()

#%% normalize

if not os.path.exists(cwd + norm_dir):
    for ID in IDs:
        print(ID)
        pt = ID.split("_")[1]
        sl = ID.split("_")[3]
        preprocess(pt,sl)


#%%

#set up wandb
with open("wandb.key" , "r") as handle:
    wandb_key = handle.readlines()[0]

wandb.login(key=wandb_key)
wandb.init(project='unet', entity='keras_krigere')


# Define the per-epoch callbacks
def log_image(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = Unet_model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=3)
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,5,1)
    plt.imshow(X_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(2,5,2)
    plt.imshow(X_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(2,5,3)
    plt.imshow(test_pred_raw[0,:,:,0])
    plt.axis("off")
    plt.title("Predicted")
    plt.subplot(2,5,4)
    plt.imshow(test_pred[0,:,:])
    plt.axis("off")
    plt.title("Predicted segmentation mask")
    plt.subplot(2,5,5)
    plt.imshow(Y_test[0,])
    plt.axis("off")
    plt.title("True")
    #plt.tight_layout()
    # Log the confusion matrix as an image summary.

    
    # Use the model to predict the values from the training dataset.
    test_pred_raw_train = Unet_model.predict(X_train_test)
    test_pred_train = np.argmax(test_pred_raw_train, axis=3)
    
    plt.subplot(2,5,6)
    plt.imshow(X_train_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(2,5,7)
    plt.imshow(X_train_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(2,5,8)
    plt.imshow(test_pred_raw_train[0,:,:,0])
    plt.axis("off")
    plt.title("Predicted")
    plt.subplot(2,5,9)
    plt.imshow(test_pred_train[0,:,:])
    plt.axis("off")
    plt.title("Predicted segmentation mask")
    plt.subplot(2,5,10)
    plt.imshow(Y_train_test[0,])
    plt.axis("off")
    plt.title("True")
    plt.tight_layout()
    
    wandb.log({"test:" : plt}, step=epoch)


# Define the per-epoch callbacks
image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)

csv_logger = CSVLogger(cwd + '/training.log')



#get training and validation data
train_IDs, val_IDs, test_IDs = get_data_split_IDs(IDs)
d_train = DataLoader(train_IDs[:30],batch_size = 2, augmentation = AUG) #len(train_IDs[:10])
d_val = DataLoader(val_IDs, batch_size = len(val_IDs)) 


#get validation data
X_val, Y_val = d_val[0]

#get a validation image with segmentation
for i in range(0,len(Y_val)):
    pixel_val_sum = np.sum(np.sum(Y_val[i,:,:])) # sum of Y_val image
    #print('i='+str(i)+' and sum='+str(pixel_val_sum))
    # find first Y_val image with sum greater than 0 (has a segmentation mask)
    if pixel_val_sum > 0:
        slice_show = i # save index value of first image with segmentation
        break

# save data for image slice with segmentation
X_test = X_val[np.newaxis,slice_show,]
Y_test = Y_val[np.newaxis,slice_show,]   



#get training image with segmentation
i = 0
while True:
    X_train, Y_train = d_train[i]
    pixel_val_sum = np.sum(np.sum(Y_train[0,:,:]))
    pixel_val_sum1 = np.sum(np.sum(Y_train[1,:,:]))# sum of Y_val image
    #print('i='+str(i)+' and sum='+str(pixel_val_sum))
    # find first Y_val image with sum greater than 0 (has a segmentation mask)
    if pixel_val_sum > 0:
        slice_show = 0 # save index value of first image with segmentation
        break
    elif pixel_val_sum1 > 0:
        slice_show = 1
        break 
    i += 1

# save data for image slice with segmentation
X_train_test = X_train[np.newaxis,slice_show,]
Y_train_test = Y_train[np.newaxis,slice_show,]    



#Create a callback that saves the model's weights every 10 epochs
BS = 2
STEPS_PER_EPOCH = np.floor (len(train_IDs) / BS)
print(STEPS_PER_EPOCH)
SAVE_PERIOD = 10

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_freq= int(SAVE_PERIOD * STEPS_PER_EPOCH))




def dice(mask_gt,mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum


class Dice_metric(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val,batch_size=2))

        # y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_dice': dice(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data


#define model
unet = Unet(512,512,2,32)
Unet_model = unet.Unet_model

#train model
Unet_model.fit(d_train, batch_size = 2, epochs=100, callbacks = [image_callback,WandbCallback(),csv_logger, cp_callback], validation_data=(X_val,Y_val), validation_batch_size = 2)


#Unet_model.save('checkpoints')


