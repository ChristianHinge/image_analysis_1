# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 20:05:36 2021

@author: simon
"""

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from skimage import io, color
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors as colors
import matplotlib.cm as cm
import os
import scipy
from dataloader_test import DataLoaderClassification, IDs, preprocess
from scipy import ndimage as nd
from skimage.measure import label
import csv

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

slice_ID = "pt_094_sl_24"


data_dir = "data/normalized"



unet_model_path = "checkpoints/0.1_step_decay_LR/model_120.hdf5"
#unet_model_path = "checkpoints/model_120.hdf5"
swap = True 

classification_model_path = "checkpoints/classification/2model_78.hdf5"



#%% load patient data 

pt = slice_ID.split("_")[1]
sl = slice_ID.split("_")[3]

# Load bone and brain slice
if swap:
    im_bone  = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
    im_brain = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
    im_seg = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
else:
    im_bone  = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
    im_brain = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
    im_seg = np.load(data_dir + f"/{pt}/seg/{sl}.npy")   
    
    
X_infer = np.zeros((1,512,512,2))
Y_seg_true = np.zeros((1,512,512))

X_infer[0,:,:,0] = im_bone
X_infer[0,:,:,1] = im_brain
Y_seg_true = im_seg

#%%


##### UNET PREDICTION #######
BS = 2

Unet_model = load_model(unet_model_path)

Y_seg_pred = np.round(Unet_model.predict(X_infer,batch_size=BS).squeeze())

#%%

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)

if swap:
    plt.imshow(X_infer[0,:,:,0])
else:
    plt.imshow(X_infer[0,:,:,1])

plt.subplot(1,2,2)
plt.imshow(Y_seg_pred[:,:])


#%%

##### CLASSIFICATION PREDICTION ######
X_class = np.zeros((1,512,512,3))
X_class[0,:,:,:2] = X_infer
X_class[0,:,:,2] = Y_seg_pred


classification_model = load_model(classification_model_path)

Y_class_pred = classification_model.predict(X_class,batch_size=BS)



#%% diagnose plot 

hem_type = ['Intraventricular','Intraparenchymal','Subarachnoid','Epidural','Subdural','No hemorrhage']

# probability
prob = np.array([0.4,0.4,0.40,0.40,0.4,0.30])
#prob = Y_class_pred
prob_plot = np.copy(prob)

# threshold
th = 0.3

if swap:
    bone_2 = X_infer[0,:,:,1]
    brain_2 = X_infer[0,:,:,0]
    true_2 = Y_seg_pred
else:
    bone_2 = X_infer[0,:,:,0]
    brain_2 = X_infer[0,:,:,1]
    true_2 = Y_seg_pred

# bone_2  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
# brain_2 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
# true_2 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")

# DETECT BLOBS
blobs = label(true_2*1 > 0)
n_blobs = len(np.unique(blobs))-1

bg = 0.3
pad_shape = ((50,50),(50,350))
# add padding
blobs_pad = np.pad(blobs,pad_shape,'minimum')
#brain_2_pad = np.pad(brain_2,((0,0),(0,100)),'minimum')     # zero padding
brain_2_pad = np.pad(brain_2,pad_shape,'constant',constant_values=((bg,bg),(bg,bg)))

# DETECT CLASSES
prob_binary = np.zeros(6)
# if max probability = no hemorrhage
if max(prob) == prob[5]:
    prob_binary[5] = 1
    print("if1")
else:
    prob[5]=0 # set no hemorrhage equal to 0
    
    # threshold
    for i in range(0,6):
        if prob[i] < th:
            prob[i] = 0
            print("if2")
            
    if n_blobs == 1:
        # max
        i = np.argmax(prob)
        prob_binary[i] = 1
        print("if 3")
    else:
        print("else 3")
        if n_blobs > np.sum(prob>0):
            for i in range(0,np.sum(prob>0)):
                i = np.argmax(prob)
                prob_binary[i] = 1
                prob[i] = 0
                print()
        elif n_blobs <= np.sum(prob>0):
            for i in range(0,n_blobs):
                i = np.argmax(prob)
                prob_binary[i] = 1
                prob[i] = 0


# plot
plt.rc('font', size=25)          # controls default text sizes
plt.figure(figsize=(12,12))
plt.imshow(brain_2_pad,cmap="gray")
plt.imshow(np.ma.masked_where(blobs_pad == 0, blobs_pad), cmap='brg', interpolation='none', alpha=0.6)


# ADD TEXT
input_text = """Brain hemorrhage findings\n
Hemorrhage type       Probability [%] """
# hem_data should be an array with length equal to number of blobs,
# containing a string of hemorrhage type labels for each blob
ax = plt.gca()
#ax.text(10, 500-i*30, hem_type[i], style='italic',color='white',fontsize=12,horizontalalignment='left',bbox={'facecolor': 'black', 'alpha': 0.5, 'pad': 10})
ax.text(590, 50, input_text, style='normal',color='white',fontsize=12,horizontalalignment='left',verticalalignment='top',weight='bold')
#ax.text(600, 50, input_text_hem, style='italic',color='white',fontsize=12,horizontalalignment='left',verticalalignment='top')

for i in range(0,len(hem_type)):
    # test if probability is over the threshold - choose text color
    input_text_blank = ".                          ."
    if prob_binary[i] > 0:
        col = 'green'
        input_text_hem = f"{hem_type[i]}"
        ax.text(600, 145+i*25, input_text_blank, style='normal',color='green',fontsize=12,horizontalalignment='left',verticalalignment='top',bbox={'facecolor': 'green', 'alpha': 1, 'pad': 5})
        ax.text(600, 145+i*25, input_text_hem, style='normal',color='white',fontsize=12,horizontalalignment='left',verticalalignment='top')

        ax.text(780, 145+i*25, f"{prob_plot[i]:.02f}", style='normal',color='white',fontsize=12,horizontalalignment='left',verticalalignment='top',weight='bold')
    else:
        col = 'white'
        input_text_hem = f"{hem_type[i]}"
        ax.text(600, 145+i*25, input_text_hem, style='normal',color='white',fontsize=12,horizontalalignment='left',verticalalignment='top')
        ax.text(780, 145+i*25, f"{prob_plot[i]:.02f}", style='normal',color=col,fontsize=12,horizontalalignment='left',verticalalignment='top')
    
# add patient data
input_text_patient = f"""Patient ID:    {pt}   \n 
Slice number:    {sl} \n
Date:    05-05-2021 """

ax.text(590, 450, input_text_patient, style='normal',color='white',fontsize=12,horizontalalignment='left',verticalalignment='top')
    
plt.axis("off")
plt.savefig("classification.jpg")



#%% BIG plot for poster 

# X_infer_plot = np.zeros((6,512,512,2))

# # X_infer_plot[0,:,:,0] = im_bone
# # X_infer_plot[0,:,:,1] = im_brain



# # load images
# # Intraventricular
# pt = 94
# sl = 20
# im_bone_c1  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
# im_brain_c1 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
# im_mask_true_c1 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
# #im_mask_pred_c1 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# X_infer_plot[0,:,:,0] = im_bone_c1
# X_infer_plot[0,:,:,1] = im_brain_c1



# # Intraparenchymal
# pt = 94
# sl = 26
# im_bone_c2  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
# im_brain_c2 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
# im_mask_true_c2 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
# #im_mask_pred_c2 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# X_infer_plot[1,:,:,0] = im_bone_c2
# X_infer_plot[1,:,:,1] = im_brain_c2



# # Subarachnoid
# # pt = 82
# # sl = 11
# pt = 76
# sl = 19
# im_bone_c3  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
# im_brain_c3 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
# im_mask_true_c3 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
# #im_mask_pred_c3 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# X_infer_plot[2,:,:,0] = im_bone_c3
# X_infer_plot[2,:,:,1] = im_brain_c3


# # Epidural
# pt = 78
# sl = 14
# im_bone_c4  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
# im_brain_c4 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
# im_mask_true_c4 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
# #im_mask_pred_c4 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# X_infer_plot[3,:,:,0] = im_bone_c4
# X_infer_plot[3,:,:,1] = im_brain_c4

# # Subdural
# pt = 81
# sl = 20
# im_bone_c5  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
# im_brain_c5 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
# im_mask_true_c5 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
# #im_mask_pred_c5 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# X_infer_plot[4,:,:,0] = im_bone_c5
# X_infer_plot[4,:,:,1] = im_brain_c5

# # no hemorrhage
# pt = 130
# sl = 19
# im_bone_c6  = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
# im_brain_c6 = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
# im_mask_true_c6 = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
# #im_mask_pred_c6 = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
  
# X_infer_plot[5,:,:,0] = im_bone_c6
# X_infer_plot[5,:,:,1] = im_brain_c6


# bone = [im_bone_c1,im_bone_c2,im_bone_c3,im_bone_c4,im_bone_c5,im_bone_c6]
# brain = [im_brain_c1,im_brain_c2,im_brain_c3,im_brain_c4,im_brain_c5,im_brain_c6]
# true = [im_mask_true_c1,im_mask_true_c2,im_mask_true_c3,im_mask_true_c4,im_mask_true_c5,im_mask_true_c6]
# #pred = [im_mask_pred_c1,im_mask_pred_c2,im_mask_pred_c3,im_mask_pred_c4,im_mask_pred_c5,im_mask_pred_c6]

# if swap:
#     X_infer2plot = np.zeros_like(X_infer_plot)
#     X_infer2plot[:,:,:,0] = X_infer_plot[:,:,:,1]
#     X_infer2plot[:,:,:,1] = X_infer_plot[:,:,:,0]
# else:
#     X_infer2plot = X_infer_plot

# pred_plot = np.round(Unet_model.predict(X_infer2plot,batch_size=BS).squeeze())

# pred = [p for p in pred_plot]

#%%
# alpha = 0.5


# hem_type = ['Intraventricular','Intraparenchymal','Subarachnoid','Epidural','Subdural','No hemorrhage']

# # plot
# plt.rc('font', size=25)          # controls default text sizes
# plt.figure(figsize=(20,24))

# cm1 = colors.ListedColormap(['red', 'blue'])
# cm2 = colors.ListedColormap(['blue','yellow'])

# xx=[180,220,320,30,130,72]
# yy=[160,120,150,220,120,20]
# shape = [250,200,150,150,330,440]

# for i in range(0,6):
#     # class 1 = no hemorrhage

#     # BONE
#     plt.subplot(6,5,1+i*5)
#     plt.imshow(bone[i],cmap="gray")
#     plt.axis("off")
#     ax = plt.gca()
#     ax.text(-0.1, 0.5, hem_type[i],horizontalalignment='right',verticalalignment='center',rotation='vertical',transform=ax.transAxes)
#     if i ==0:
#         plt.title("Brain \n")
#     # BRAIN
#     plt.subplot(6,5,2+i*5)
#     plt.imshow(brain[i],cmap="gray")
#     #plt.imshow(im_mask[:,:],cmap="gray",alpha=0.5)
#     plt.axis("off")
#     if i ==0:
#         plt.title("Bone \n")
#     # BRAIN + TRUE MASK
#     plt.subplot(6,5,3+i*5)
#     plt.imshow(brain[i],cmap="gray")
#     plt.imshow(np.ma.masked_where(true[i] == 0, true[i]), cmap="brg", interpolation='none', alpha=0.5)
#     plt.axis("off")
#     if i ==0:
#         plt.title("Brain \n true mask")
#     plt.tight_layout()
#     # BRAIN + PREDICTED MASK
#     plt.subplot(6,5,4+i*5)
#     plt.imshow(brain[i],cmap="gray")
#     plt.imshow(np.ma.masked_where(pred[i] == 0, pred[i]), cmap="hsv", interpolation='none', alpha=0.5)
#     plt.axis("off")
#     if i ==0:
#         plt.title("Brain \n predicted mask")
#     # BRAIN + TRUE MASK + PREDICTED MASK [ZOOM]
#     mix = (true[i]*1+pred[i]*1.5)/2
#     plt.subplot(6,5,5+i*5)
#     plt.imshow(brain[i],cmap="gray")
#     #plt.imshow(mix,cmap="gray")
#     #plt.imshow(np.ma.masked_where(true == 0, true == 1), cmap="brg", interpolation='none', alpha=0.5)
#     #plt.imshow(np.ma.masked_where(mix < 0.5, mix), cmap="brg", interpolation='none', alpha=0.5)
#     plt.imshow(np.ma.masked_where(pred[i] == 0, pred[i]), cmap=cm1, interpolation='none', alpha=0.6)
#     plt.imshow(np.ma.masked_where(true[i] == 0, true[i]), cmap=cm2, interpolation='none', alpha=0.6)
    
#     plt.xlim((xx[i],xx[i]+shape[i]))
#     plt.ylim((yy[i]+shape[i],yy[i]))
#     plt.axis("off")
#     if i ==0:
#         plt.title("Mask \n true+predicted")

# plt.tight_layout()
# plt.savefig("hej2.jpg")


