# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:23:01 2021

@author: kdamc
"""

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

data_dir = "data/normalized"

#%%
        
hem_data = []                # load hemorrhage diagnosis data
with open('data/hemorrhage_diagnosis.csv', newline='') as File:  
    reader = csv.reader(File)
    for row in reader:
        hem_data.append(np.array(row))


hem_data = np.array(hem_data)

rows, cols = hem_data.shape

# remove fracture data (last column)
hem_data = np.delete(hem_data,cols-1,1)
# remove table labels
hem_data = hem_data[1:,:].astype(int)
# we now have:
# patient_ID, slice number, intraventricular, intraparenchymal, subarachnoid, epidural, subdural, no hemorrhage
# remove row 1098 - error
hem_data = np.delete(hem_data,1098,0)

# update shape
rows, cols = hem_data.shape

#%%

np.unique(hem_data[np.where(hem_data[:,7] == 1),0])
hem_data[np.where(hem_data[:,7] == 1),1]


#%%
data_dir = "data/Patients_CT"
norm_dir = "/data/normalized"

cwd = os.getcwd()
IDs_slice = ["pt_094_sl_26","pt_094_sl_20","pt_130_sl_19","pt_078_sl_14","pt_081_sl_20","pt_082_sl_11"     , "pt_094_sl_24"      , "pt_082_sl_12","pt_094_sl_21", "pt_094_sl_27","pt_078_sl_15","pt_081_sl_21"]
if not os.path.exists(cwd + norm_dir):
    for ID in IDs_slice:
        print(ID)
        pt = ID.split("_")[1]
        sl = ID.split("_")[3]
        preprocess(pt,sl)
        
#%%
data_dir = "data/normalized"

# load images
# Intraventricular
pt = 94
sl = 20
im_bone_c1  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
im_brain_c1 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
im_mask_true_c1 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
im_mask_pred_c1 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# Intraparenchymal
pt = 94
sl = 26
im_bone_c2  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
im_brain_c2 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
im_mask_true_c2 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
im_mask_pred_c2 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# Subarachnoid
pt = 82
sl = 11
im_bone_c3  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
im_brain_c3 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
im_mask_true_c3 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
im_mask_pred_c3 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# Epidural
pt = 78
sl = 14
im_bone_c4  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
im_brain_c4 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
im_mask_true_c4 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
im_mask_pred_c4 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# Subdural
pt = 81
sl = 20
im_bone_c5  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
im_brain_c5 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
im_mask_true_c5 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")
im_mask_pred_c5 = np.load(data_dir + f"/0{pt}/seg/{sl+1}.npy")

# no hemorrhage
pt = 130
sl = 19
im_bone_c6  = np.load(data_dir + f"/{pt}/bone/{sl}.npy")
im_brain_c6 = np.load(data_dir + f"/{pt}/brain/{sl}.npy")
im_mask_true_c6 = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
im_mask_pred_c6 = np.load(data_dir + f"/{pt}/seg/{sl}.npy")
  
bone = [im_bone_c1,im_bone_c2,im_bone_c3,im_bone_c4,im_bone_c5,im_bone_c6]
brain = [im_brain_c1,im_brain_c2,im_brain_c3,im_brain_c4,im_brain_c5,im_brain_c6]
true = [im_mask_true_c1,im_mask_true_c2,im_mask_true_c3,im_mask_true_c4,im_mask_true_c5,im_mask_true_c6]
pred = [im_mask_pred_c1,im_mask_pred_c2,im_mask_pred_c3,im_mask_pred_c4,im_mask_pred_c5,im_mask_pred_c6]


#%%
alpha = 0.5


hem_type = ['Intraventricular','Intraparenchymal','Subarachnoid','Epidural','Subdural','No hemorrhage']

# plot
plt.rc('font', size=25)          # controls default text sizes
plt.figure(figsize=(20,24))

cm1 = colors.ListedColormap(['red', 'blue'])
cm2 = colors.ListedColormap(['blue','yellow'])

xx=[180,220,50,30,130,72]
yy=[160,120,180,220,120,20]
shape = [250,200,200,150,330,440]

for i in range(0,6):
    # class 1 = no hemorrhage

    # BONE
    plt.subplot(6,5,1+i*5)
    plt.imshow(bone[i],cmap="gray")
    plt.axis("off")
    ax = plt.gca()
    ax.text(-0.1, 0.5, hem_type[i],horizontalalignment='right',verticalalignment='center',rotation='vertical',transform=ax.transAxes)
    if i ==0:
        plt.title("Brain \n")
    # BRAIN
    plt.subplot(6,5,2+i*5)
    plt.imshow(brain[i],cmap="gray")
    #plt.imshow(im_mask[:,:],cmap="gray",alpha=0.5)
    plt.axis("off")
    if i ==0:
        plt.title("Bone \n")
    # BRAIN + TRUE MASK
    plt.subplot(6,5,3+i*5)
    plt.imshow(brain[i],cmap="gray")
    plt.imshow(np.ma.masked_where(true[i] == 0, true[i]), cmap="brg", interpolation='none', alpha=0.5)
    plt.axis("off")
    if i ==0:
        plt.title("Brain \n true mask")
    plt.tight_layout()
    # BRAIN + PREDICTED MASK
    plt.subplot(6,5,4+i*5)
    plt.imshow(brain[i],cmap="gray")
    plt.imshow(np.ma.masked_where(pred[i] == 0, pred[i]), cmap="hsv", interpolation='none', alpha=0.5)
    plt.axis("off")
    if i ==0:
        plt.title("Brain \n predicted mask")
    # BRAIN + TRUE MASK + PREDICTED MASK [ZOOM]
    mix = (true[i]*1+pred[i]*1.5)/2
    plt.subplot(6,5,5+i*5)
    plt.imshow(brain[i],cmap="gray")
    #plt.imshow(mix,cmap="gray")
    #plt.imshow(np.ma.masked_where(true == 0, true == 1), cmap="brg", interpolation='none', alpha=0.5)
    #plt.imshow(np.ma.masked_where(mix < 0.5, mix), cmap="brg", interpolation='none', alpha=0.5)
    plt.imshow(np.ma.masked_where(pred[i] == 0, pred[i]), cmap=cm1, interpolation='none', alpha=0.6)
    plt.imshow(np.ma.masked_where(true[i] == 0, true[i]), cmap=cm2, interpolation='none', alpha=0.6)
    
    plt.xlim((xx[i],xx[i]+shape[i]))
    plt.ylim((yy[i]+shape[i],yy[i]))
    plt.axis("off")
    if i ==0:
        plt.title("Mask \n true+predicted")

plt.tight_layout()
plt.savefig("hej.jpg")



#%% PLOT 2

# INSERT PARAMENTERS
# patient ID
pt = 94
# slice
sl = 49
# probability
prob = np.array([0.4,0.4,0.40,0.40,0.4,0.30])
prob_plot = np.copy(prob)
# threshold
th = 0.3

data_dir = "data/normalized"

pt = 94
sl = 24
bone_2  = np.load(data_dir + f"/0{pt}/bone/{sl}.npy")
brain_2 = np.load(data_dir + f"/0{pt}/brain/{sl}.npy")
true_2 = np.load(data_dir + f"/0{pt}/seg/{sl}.npy")

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

#prob_binary = [0,1,0,0,1,1]



# plot
plt.rc('font', size=25)          # controls default text sizes
plt.figure(figsize=(12,12))
plt.imshow(brain_2_pad,cmap="gray")
plt.imshow(np.ma.masked_where(blobs_pad == 0, blobs_pad), cmap='brg', interpolation='none', alpha=0.6)


# ADD TEXT
input_text = """Brain hemorrhage \n
Segmentaiton and classification  \n \n 
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

#dpi
# io.imshow(color.label2rgb(im_mask,im_bone,colors=[(255,0,0)],alpha=0.01, bg_label=0, bg_color=None))



#%% PLOT - EXTRA

#plt.imshow(np.ma.masked_where(pred_2 == 0, pred_2), cmap='brg', interpolation='none', alpha=0.6)
plt.imshow(np.ma.masked_where(true_2 == 0, true_2), cmap='brg', interpolation='none', alpha=0.6)
    
plt.xlim((xx[i],xx[i]+shape[i]))
plt.ylim((yy[i]+shape[i],yy[i]))
plt.axis("off")
