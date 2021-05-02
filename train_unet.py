# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:14:51 2021

@author: simon
"""
#%%
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from Unet_works import Unet
from dataloader_test import DataLoader, IDs, preprocess, load_train_val_data
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

#%% preprocess data 

if not os.path.exists(cwd + norm_dir):
    for ID in IDs:
        print(ID)
        pt = ID.split("_")[1]
        sl = ID.split("_")[3]
        preprocess(pt,sl)


#%%

train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val, X_val_test, Y_val_test, X_train_test, Y_train_test, X_test, Y_test = load_train_val_data()


#set up wandb
with open("wandb.key" , "r") as handle:
    wandb_key = handle.readlines()[0]

wandb.login(key=wandb_key)
wandb.init(project='unet', entity='keras_krigere')


# Define the per-epoch callbacks
def log_image(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = Unet_model.predict(X_val_test)
    test_pred = np.round(test_pred_raw)
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,5,1)
    plt.imshow(X_val_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(2,5,2)
    plt.imshow(X_val_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(2,5,3)
    plt.imshow(test_pred_raw[0,:,:,0])
    plt.axis("off")
    plt.title("Predicted")
    plt.colorbar()
    plt.subplot(2,5,4)
    plt.imshow(test_pred[0,:,:])
    plt.axis("off")
    plt.title("Predicted segmentation mask")
    plt.subplot(2,5,5)
    plt.imshow(Y_val_test[0,])
    plt.axis("off")
    plt.title("True")
    #plt.tight_layout()

    
    # Use the model to predict the values from the training dataset.
    test_pred_raw_train = Unet_model.predict(X_train_test)
    test_pred_train = np.round(test_pred_raw_train)
    
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
    plt.colorbar()
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



def scheduler(epoch, lr):
    print(lr)
    if epoch == 80:
      return lr * 0.1
    else:
      return lr

LR_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



# Define the per-epoch callbacks
image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)

csv_logger = CSVLogger(cwd + '/training.log')


#Create a callback that saves the model every 10 epochs
BS = 2
STEPS_PER_EPOCH = np.floor (len(train_IDs) / BS)
SAVE_MODEL_EPOCHS = 10
#DECAY_LR_EPOCHS = 80
#decay_steps = int(DECAY_LR_EPOCHS * STEPS_PER_EPOCH)


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_freq= int(SAVE_MODEL_EPOCHS * STEPS_PER_EPOCH))



#define model
unet = Unet(512,512,2,32)
Unet_model = unet.Unet_model


#train model
Unet_model.fit(d_train, batch_size = BS, epochs=120, callbacks = [image_callback, WandbCallback(), csv_logger, cp_callback, LR_callback], validation_data=(X_val,Y_val), validation_batch_size = BS)

Unet_model.save('Final_model')




#%%

# def dice(mask_gt,mask_pred):
#     volume_sum = mask_gt.sum() + mask_pred.sum()
#     if volume_sum == 0:
#         return np.NaN
#     volume_intersect = (mask_gt & mask_pred).sum()
#     return 2*volume_intersect / volume_sum

# def dice_loss(y_true, y_pred):
#   y_true = tf.cast(y_true, tf.float32)
#   y_pred = tf.math.sigmoid(y_pred)
#   numerator = 2 * tf.reduce_sum(y_true * y_pred)
#   denominator = tf.reduce_sum(y_true + y_pred)

#   return 1 - numerator / denominator





# class Dice_metric(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self._data = []

#     def on_epoch_end(self, batch, logs={}):
#         X_val, y_val = self.validation_data[0], self.validation_data[1]
#         y_predict = np.asarray(self.model.predict(X_val,batch_size=2))

#         # y_val = np.argmax(y_val, axis=1)
#         y_predict = np.round(y_predict)
        
        
#         self._data.append({
#             'val_dice': dice(y_val, y_predict),
#         })
#         return

#     def get_data(self):
#         return self._data




# class dice_metric(keras.metrics.Metric):
#     def __init__(self, name="dice_metric", **kwargs):
#         super(dice_metric, self).__init__(name=name, **kwargs)
#         self.dice = self.add_weight(name="dice", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         print(y_true.shape)
#         print(y_pred.shape)
#         y_pred = tf.reshape(tf.argmax(y_pred,axis=0), shape=(1,-1))
#         # y_pred = tf.dtypes.cast(y_pred, tf.float32)
#         # y_true = tf.dtypes.cast(y_true, tf.float32)

#         #volume_sum = y_true.sum() + y_pred.sum()
#         volume_sum = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred)
#         if volume_sum == 0:
#             return np.NaN
#         #volume_intersect = (y_true & y_pred).sum()
#         volume_intersect = tf.math.reduce_sum(y_true & y_pred)
#         self.dice.assign(2*volume_intersect / volume_sum)

#     def result(self):
#         return self.dice

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.dice.assign(0.0)
