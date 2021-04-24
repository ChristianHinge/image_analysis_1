
#%%

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
import cv2

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
    batch_size=32, dim=(512,512),n_channels=2,n_classes=6,shuffle=True,augmentation = None):

        #Create the dataloader
        self.list_IDs = list_IDs #list of patient IDs (1,2,3,4,5,...)
        self.to_fit = to_fit     #True: dataloader used for training. False: dataloader used for test/val
        self.batch_size = batch_size #Size of bateches
        self.dim = dim           #Image dimension (650,650)
        self.n_channels = n_channels #Channels, 2: Brain, Bone
        self.n_classes = n_classes #Number of classes in segmentation, 6: intraventricular, intraparenchymal, subarachnoid, epidural, subdural, no hemorrhage
        self.shuffle = shuffle #Randomize the order of the dataset
        self.augmentation = augmentation #Augmentation function used when training
        self.data_dir = "data/normalized" #Data directory
        self.on_epoch_end()

                # load hemorrhage diagnosis data
        with open('data/hemorrhage_diagnosis.csv', newline='') as File:  
            reader = csv.reader(File)
            for row in reader:
                hem_data.append(np.array(row))


        hem_data = np.array(hem_data)

        rows, cols = hem_data.shape

        # remove fracture data (last column)
        hem_data = np.delete(hem_data,cols-1,1)
        # we now have:
        # patient_ID, slice number, intraventricular, intraparenchymal, subarachnoid, epidural, subdural, no hemorrhage

        # update shape
        rows, cols = hem_data.shape
        self.hem_data = hem_data
        self.Y_dim = self.cols

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
            Y = np.zeros((len(IDs),self.Y_dim))

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
                Y[i,:] = self.hem_data[hem_data[:,0]==]

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


#%%

def preprocess(pt,sl):
    """
    preprocessing
    """

    data_dir = "data/Patients_CT"
    out_dir = "data/normalized"

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

    path1 = out_dir+f"/{pt}/brain/"
    path2 = out_dir+f"/{pt}/bone/"
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
    im2 = (im-im.mean())/np.std(im)
    return im2

def normalize2(im2):
    im2 = im2 != 0
    return im2



#%%

ixs = list(range(49,131))
IDs = []

for i, ix in enumerate(ixs):
    f = f"data/Patients_CT/{ix:03d}/bone"
    n_slices = len(os.listdir(f))
    IDs.extend([f"pt_{ix:03d}_sl_{i}" for i in range(1,n_slices+1)])
#%%""
"""
for ID in IDs:
    print(ID)
    pt = ID.split("_")[1]
    sl = ID.split("_")[3]
    preprocess(pt,sl)
"""
# %%
