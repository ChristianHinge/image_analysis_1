
#%%

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
import cv2
import csv
from data_split import get_data_split_IDs
from test2 import AUG

#%% Dataloader

#dataloader[0] #returnerer batch 1
#len(dataloader) #returnerer antal observationer / batch_size

# Dataloader class based on TF
class DataLoader(Sequence):

    def __init__(self,list_IDs,to_fit=True, 
    batch_size=32, dim=(512,512),n_channels=2,n_classes=2,shuffle=True,augmentation = None):

        #Create the dataloader
        self.list_IDs = list_IDs #list of patient IDs (1,2,3,4,5,...)
        self.to_fit = to_fit     #True: dataloader used for training. False: dataloader used for test/val
        self.batch_size = batch_size #Size of bateches
        self.dim = dim           #Image dimension (650,650)
        self.n_channels = n_channels #Channels, 2: Brain, Bone
        self.n_classes = n_classes #Number of classes in segmentation, 2: hemorrage, no hemorrage
        self.shuffle = shuffle #Randomize the order of the dataset
        self.augmentation = augmentation #Augmentation function used when training
        self.data_dir = "data/normalized" #Data directory
        self.on_epoch_end()

    # Returns the length of the dataset in terms of number of batches
    def __len__(self):
        return len(self.list_IDs)//self.batch_size

    # Is run at every epoch end. Randomizes order of dataset
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    # Function to get a batch given a batch index
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_IDs = [self.list_IDs[ix] for ix in indexes]

        return self._load_batch(batch_IDs,load_Y=self.to_fit)


    # Function to load a batch, (apply augmentation), and return the batch
    # load_Y = False when doing testing/validation
    def _load_batch(self, IDs, load_Y = True):
        # Initialize batch arrays
        X = np.zeros((len(IDs),*self.dim, self.n_channels))
        if load_Y:
            Y = np.zeros((len(IDs),*self.dim))

        for i, ID in enumerate(IDs):
            pt = ID.split("_")[1]
            sl = ID.split("_")[3]

             # Load bone and brain slice
            im_bone  = np.load(self.data_dir + f"/{pt}/bone/{sl}.npy")
            im_brain = np.load(self.data_dir + f"/{pt}/brain/{sl}.npy")

            X[i,:,:,0] = im_bone
            X[i,:,:,1] = im_brain

            # Load segmentation mask if training
            if load_Y:
                Y[i,:,:] = np.load(self.data_dir + f"/{pt}/seg/{sl}.npy")

                # Apply augmentation
                if self.augmentation != None:
                    
                    for i in range(self.batch_size):
                        X_, Y_ = self.augmentation(X[i,], Y[i,])
                        X[i,] = X_
                        Y[i,] = Y_

        # Return X and Y if training
        if load_Y:        
            return X, Y
        else:
            # Return X if not training
            return X


class DataLoaderClassification(Sequence):

    def __init__(self,list_IDs,to_fit=True, 
        batch_size=32, dim=(512,512),n_channels=3,n_classes=6,shuffle=True,augmentation = None,data_dir="data/normalized"):

        #Create the dataloader
        self.list_IDs = list_IDs     #list of patient IDs (1,2,3,4,5,...)
        self.to_fit = to_fit         #True: dataloader used for training. False: dataloader used for test/val
        self.batch_size = batch_size #Size of batches
        self.dim = dim               #Image dimension (650,650)
        self.n_channels = n_channels #Channels, 2: Brain, Bone
        self.n_classes = n_classes   #Number of classes in segmentation, 6: intraventricular, intraparenchymal, subarachnoid, epidural, subdural, no hemorrhage
        self.shuffle = shuffle       #Randomize the order of the dataset
        self.augmentation = augmentation #Augmentation function used when training
        self.data_dir = data_dir #Data directory
        self.on_epoch_end()

        hem_data = []                # load hemorrhage diagnosis data
        with open('data/hemorrhage_diagnosis.csv', newline='') as File:  
            reader = csv.reader(File)
            for row in reader:
                hem_data.append(np.array(row))


        hem_data = np.array(hem_data)

        rows, cols = hem_data.shape

        # remove fracture data (last column)
        hem_data = np.delete(hem_data,cols-1,1)
        # remove labels
        hem_data = hem_data[1:,:].astype(int)
        # we now have:
        # patient_ID, slice number, intraventricular, intraparenchymal, subarachnoid, epidural, subdural, no hemorrhage

        # remove row 1098 - error
        hem_data = np.delete(hem_data,1098,0)
        
        # update shape
        rows, cols = hem_data.shape
        self.hem_data = hem_data
        self.Y_dim = cols

    # Returns the length of the dataset in terms of number of batches
    def __len__(self):
        return len(self.list_IDs)//self.batch_size

    # Is run at every epoch end. Randomizes order of dataset
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    # Function to get a batch given a batch index
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_IDs = [self.list_IDs[ix] for ix in indexes]
        return self._load_batch(batch_IDs,load_Y=self.to_fit)

    # Function to load a batch, (apply augmentation), and return the batch
    # load_Y = False when doing testing/validation
    def _load_batch(self, IDs, load_Y = True):
        # Initialize batch arrays
        X = np.zeros((len(IDs),*self.dim, self.n_channels))
        if load_Y:
            Y = np.zeros((len(IDs),self.Y_dim-2))

        for i, ID in enumerate(IDs):
            pt = ID.split("_")[1]
            sl = ID.split("_")[3]

             # Load bone and brain slice
            im_bone  = np.load(self.data_dir + f"/{pt}/bone/{sl}.npy")
            im_brain = np.load(self.data_dir + f"/{pt}/brain/{sl}.npy")
            im_seg = np.load(self.data_dir + f"/{pt}/seg/{sl}.npy")
            
            X[i,:,:,0] = im_bone
            X[i,:,:,1] = im_brain
            X[i,:,:,2] = im_seg

            # Load segmentation mask if training
            if load_Y:
                ix = (self.hem_data[:,0]==int(pt)) & (self.hem_data[:,1] == int(sl))
                Y[i,:] = self.hem_data[ix,2:]

                # Apply augmentation
                if self.augmentation != None:
                    for i in range(self.batch_size):
                        X_, X2_ = self.augmentation(X[i,:,:,:2], X[i,:,:,2])
                        X[i,:,:,:2] = X_
                        X[i,:,:,2] = X2_


        # Return X and Y if training
        if load_Y:        
            return X, Y
        else:
            # Return X if not training
            return X
#%%

def preprocess(pt,sl,data_dir = "data/normalized"):
    """
    preprocessing
    """
    out_dir = data_dir
    data_dir = "data/Patients_CT"

    

    im_bone = np.array(Image.open(data_dir + f"/{pt}/bone/{sl}.jpg"))
    im_brain = np.array(Image.open(data_dir + f"/{pt}/brain/{sl}.jpg"))
    seg_path = data_dir + f"/{pt}/brain/{sl}_HGE_Seg.jpg"

    if os.path.exists(seg_path):
        seg = np.array(Image.open(seg_path))
    else:
        seg = np.zeros(im_bone.shape)

    seg = cv2.resize(seg, (512,512), interpolation = cv2.INTER_CUBIC)
    im_bone = cv2.resize(im_bone, (512,512), interpolation = cv2.INTER_CUBIC)
    im_brain = cv2.resize(im_brain, (512,512), interpolation = cv2.INTER_CUBIC)

    path1 = out_dir+f"/{pt}/bone/"
    path2 = out_dir+f"/{pt}/brain/"
    path3 = out_dir+f"/{pt}/seg/"

    if not os.path.exists(path1):
        os.makedirs(path1)

    if not os.path.exists(path2):
        os.makedirs(path2)

    if not os.path.exists(path3):
        os.makedirs(path3)

    im1 = normalize1(im_bone)
    im2 = normalize1(im_brain)
    seg = normalize2(seg)

    np.save(path1+f"{sl}.npy",im1)
    np.save(path2+f"{sl}.npy",im2)
    np.save(path3+f"{sl}.npy",seg)


def normalize1(im):
<<<<<<< HEAD
    #im2 = (im-im.mean())/np.std(im)
    im = im/255
    return im
=======
    
    im2 = (im-im.mean())/np.std(im)
    return im2
>>>>>>> 5aec86455db91453c75904e4c911cd09b5d30166

def normalize2(im2):
    
    im2 = im2 > 100
    return im2
"""
#%%
train_ids, val_ids, test_ids = get_data_split_IDs()


for ID in train_ids + test_ids + val_ids:
    print(ID)
    pt = ID.split("_")[1]
    sl = ID.split("_")[3]
    preprocess(pt,sl)

# %%
"""
def load_train_val_data():
    #get training and validation data
    train_IDs, val_IDs, test_IDs = get_data_split_IDs()
    d_train = DataLoader(train_IDs,batch_size = 2, augmentation = AUG)
    d_val = DataLoader(val_IDs, batch_size = len(val_IDs), shuffle=False) 
    d_test = DataLoader(test_IDs, batch_size = len(test_IDs), shuffle=False)
    
    #get validation data
    X_val, Y_val = d_val[0]
    
    #get validation data
    X_test, Y_test = d_test[0]
    
    #get a validation image with segmentation
    for i in range(0,len(Y_val)):
        pixel_val_sum = np.sum(np.sum(Y_val[i,:,:])) # sum of Y_val image
        #print('i='+str(i)+' and sum='+str(pixel_val_sum))
        # find first Y_val image with sum greater than 0 (has a segmentation mask)
        if pixel_val_sum > 0:
            slice_show = i # save index value of first image with segmentation
            break
    
    # save data for image slice with segmentation
    X_val_test = X_val[np.newaxis,slice_show,]
    Y_val_test = Y_val[np.newaxis,slice_show,]   
    
    
    
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
    
    return train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val, X_val_test, Y_val_test, X_train_test, Y_train_test, X_test, Y_test


def load_train_val_data_classifier():
    #get training and validation data
    train_IDs, val_IDs, test_IDs = get_data_split_IDs()
<<<<<<< HEAD
    d_train = DataLoaderClassification(train_IDs,batch_size = 4, augmentation = None) #len(train_IDs[:10])
    d_val = DataLoaderClassification(val_IDs, batch_size = len(val_IDs)) 
=======
    d_train = DataLoaderClassification(train_IDs,batch_size = 2, augmentation = AUG) #len(train_IDs[:10])
    d_val = DataLoaderClassification(val_IDs, batch_size = len(val_IDs), shuffle=False) 
>>>>>>> 5aec86455db91453c75904e4c911cd09b5d30166
    
    
    #get validation data
    X_val, Y_val = d_val[0]
    
    
    #get a validation image with segmentation
    for i in range(0,len(X_val)):
        pixel_val_sum = np.sum(np.sum(X_val[i,:,:,2])) # sum of Y_val image
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
        pixel_val_sum = np.sum(np.sum(X_train[0,:,:,2]))
        pixel_val_sum1 = np.sum(np.sum(X_train[1,:,:,2]))# sum of Y_val image
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
    
    return train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val, X_test, Y_test, X_train_test, Y_train_test