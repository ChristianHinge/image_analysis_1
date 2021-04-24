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

cwd = os.getcwd()

#%%

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
    
    plt.figure(figsize=(10,3))
    plt.subplot(1,5,1)
    plt.imshow(X_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(1,5,2)
    plt.imshow(X_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(1,5,3)
    plt.imshow(test_pred_raw[0,:,:,0])
    plt.axis("off")
    plt.title("Predicted")
    plt.subplot(1,5,4)
    plt.imshow(test_pred[0,:,:])
    plt.axis("off")
    plt.title("Predicted segmentation mask")
    plt.subplot(1,5,5)
    plt.imshow(Y_test[0,])
    plt.axis("off")
    plt.title("True")
    plt.tight_layout()
    # Log the confusion matrix as an image summary.

    wandb.log({"test:" : plt}, step=epoch)


# Define the per-epoch callback.
image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)

csv_logger = CSVLogger(cwd + '/training.log')



#get training and validation data
train_IDs, val_IDs, test_IDs = get_data_split_IDs(IDs)
d_train = DataLoader(train_IDs,batch_size = 2) #len(train_IDs[:10])
d_val = DataLoader(val_IDs, batch_size = len(val_IDs)) 


#get validation data
X_val, Y_val = d_val[0]

#prepare data for on_epoch_end images
X_test = X_val[np.newaxis,-44,]
Y_test = Y_val[np.newaxis,-44,]


#define model
unet = Unet(512,512,2,32)
Unet_model = unet.Unet_model

#train model
Unet_model.fit(d_train, batch_size = 2, epochs=50, callbacks = [image_callback,WandbCallback(),csv_logger], validation_data=(X_val,Y_val), validation_batch_size = 2)

Unet_model.save('path/to/location')


