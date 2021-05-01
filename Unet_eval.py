# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:07:27 2021

@author: simon
"""
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf 
from keras import backend as K
import numpy as np 
import matplotlib.pyplot as plt
from dataloader_test import DataLoader, IDs, load_train_val_data
from keras.models import load_model
from tensorflow.python.client import device_lib
from scipy.spatial.distance import dice
from scipy.spatial.distance import jaccard
from Unet_works import Unet

print(device_lib.list_local_devices())

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%

#model_path = "checkpoints/model_120.hdf5"
#model_path = "Final_model"
model_path = "checkpoints/first low LR model/model_120.hdf5"
#model_path = "checkpoints/0.1_step_decay_LR/model_120.hdf5"

train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val, X_val_test, Y_val_test, X_train_test, Y_train_test, X_test, Y_test = load_train_val_data()

eval_IDs = test_IDs
X_eval = X_test
Y_eval_true = Y_test


#%%
data_dir = "data/normalized"

X_eval_imgs = np.zeros([len(eval_IDs),512,512,2])
Y_eval_imgs = np.zeros([len(eval_IDs),512,512])


for i, ID in enumerate(eval_IDs):
    pt = ID.split("_")[1]
    sl = ID.split("_")[3]

      # Load bone and brain slice
    im_bone  = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
    im_brain = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
    im_seg = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
    
    X_eval_imgs[i,:,:,0] = im_bone
    X_eval_imgs[i,:,:,1] = im_brain
    Y_eval_imgs[i,:,:] = im_seg.squeeze()


diagnosis_table = np.genfromtxt('data/hemorrhage_diagnosis.csv',delimiter=',')
diagnosis_table = diagnosis_table[1:,:]
diagnosis_table = np.delete(diagnosis_table,1098,0)

hem_binary = diagnosis_table[:,-2]

eval_IDs_hem_bool = []

for ii,ID in enumerate(eval_IDs):

    pt = ID.split("_")[1]
    sl = ID.split("_")[3]
    
    if hem_binary[(diagnosis_table[:,0] == int(pt)) & (diagnosis_table[:,1] == int(sl)) ] == 0:
        eval_IDs_hem_bool.append(True)
    else:
        eval_IDs_hem_bool.append(False)


X_eval_imgs_hem = X_eval_imgs[eval_IDs_hem_bool,:,:,:]
Y_eval_imgs_hem = Y_eval_imgs[eval_IDs_hem_bool,:,:]


#%% load model and make prediction 
 
BS = 2

Unet_model = load_model(model_path)

#Y_eval_pred_imgs = np.round(Unet_model.predict(X_eval_imgs,batch_size=BS).squeeze())
Y_eval_pred = np.round(Unet_model.predict(X_eval,batch_size=BS).squeeze())


#plot some predictions from the whole evaluation set
idx = 22

plt.subplot(2,3,4)
plt.imshow(X_test[idx,:,:,1])
plt.subplot(2,3,5)
plt.imshow(Y_test[idx,:,:])
plt.subplot(2,3,6)
plt.imshow(Y_eval_pred[idx,:,:])


#%% make predictions on ONLY hemorrhage slices from evaluation set

BS = 2

Y_eval_pred_hem = np.round(Unet_model.predict(X_eval_imgs_hem,batch_size=BS).squeeze())

Y_eval_true_hem = Y_eval_imgs_hem


#plot some predictions from ONLY hemorrhage images in evaluation set
idx = 7

plt.figure(figsize=(10,6))
plt.subplot(1,3,1)
plt.imshow(X_eval_imgs_hem[idx,:,:,0])
plt.subplot(1,3,2)
plt.imshow(Y_eval_true_hem[idx,:,:])
plt.subplot(1,3,3)
plt.imshow(Y_eval_pred_hem[idx,:,:])


#%%


def dice_coef(y_true, y_pred, smooth=1):
    
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    volume_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    if volume_sum!=0:
        smooth = 0
        
    return (2. * intersection + smooth) / (volume_sum + smooth)


def IoU_coef(y_true, y_pred,smooth=1):
    
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    if union!=0:
        smooth = 0

    return (intersection + smooth) / (union + smooth)


#%% Compute metrics for whole evaluation set

DSC = []
IoU = []

BS = 1

for b in range(int(Y_eval_true.shape[0]/BS)):
    
    bb = b*BS

    Y_eval_true_batch = Y_eval_true[bb:bb+BS,:,:]
    Y_eval_pred_batch = Y_eval_pred[bb:bb+BS,:,:]
    
    DSC_batch = dice_coef(tf.convert_to_tensor(Y_eval_true_batch), tf.convert_to_tensor(Y_eval_pred_batch))
    IoU_batch = IoU_coef(tf.convert_to_tensor(Y_eval_true_batch), tf.convert_to_tensor(Y_eval_pred_batch))
    
    # tf.print(DSC_batch)
    # tf.print(IoU_batch)
    
    DSC.append(DSC_batch)
    IoU.append(IoU_batch)
    

print('Mean Dice: {0}'.format(np.array(DSC).mean()))
print('Mean IoU: {0}'.format(np.array(IoU).mean()))


#%% Compute metrics for ONLY hemorrhage part of evaluation set

DSC_hem = []
IoU_hem = []

BS = 1

for b in range(int(Y_eval_true_hem.shape[0]/BS)):
    
    bb = b*BS

    Y_eval_true_batch = Y_eval_true_hem[bb:bb+BS,:,:]
    Y_eval_pred_batch = Y_eval_pred_hem[bb:bb+BS,:,:]
    
    DSC_batch = dice_coef(tf.convert_to_tensor(Y_eval_true_batch), tf.convert_to_tensor(Y_eval_pred_batch))
    IoU_batch = IoU_coef(tf.convert_to_tensor(Y_eval_true_batch), tf.convert_to_tensor(Y_eval_pred_batch))
    
    # tf.print(DSC_batch)
    # tf.print(IoU_batch)
    
    DSC_hem.append(DSC_batch)
    IoU_hem.append(IoU_batch)

print('Mean Dice on hem: {0}'.format(np.array(DSC_hem).mean()))
print('Mean IoU on hem: {0}'.format(np.array(IoU_hem).mean()))



#%%

# def dice_coef2(mask_gt,mask_pred, smooth=1):
#     volume_sum = mask_gt.sum() + mask_pred.sum()    
#     mask_gt_f = mask_gt.flatten()
#     mask_pred_f = mask_pred.flatten()
    
#     volume_intersect = (mask_gt_f * mask_pred_f).sum()
    
#     return (2*volume_intersect + smooth) / (volume_sum + smooth)




