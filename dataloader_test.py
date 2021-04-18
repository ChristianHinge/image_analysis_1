
#%%

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os

#%% Dataloader


#%%
# Dataloader class based on TF
class DataLoader(Sequence):

    def __init__(self,list_IDs,to_fit=True, 
    batch_size=32, dim=(650,650),n_channels=2,n_classes=2,shuffle=True,augmentation = None):

        #Create the dataloader
        self.list_IDs = list_IDs #list of patient IDs (1,2,3,4,5,...)
        self.to_fit = to_fit #True: dataloader used for training. False: dataloader used for test/val
        self.batch_size = batch_size #Size of bateches
        self.dim = dim #Image dimension (650,650)
        self.n_channels = n_channels #CHannels, 2: Brain, Bone
        self.n_classes = n_classes #Number of classes in segmentation, 2: hemorrage, no hemorrage
        self.shuffle = shuffle #Randomize the order of the dataset
        self.augmentation = augmentation #Augmentation function used when training
        self.data_dir = "data/Patients_CT" #Data directory

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
            im_bone  = Image.open(self.data_dir + f"/{pt}/bone/{sl}.jpg")
            im_brain = Image.open(self.data_dir + f"/{pt}/brain/{sl}.jpg")
    
            X[i,:,:,0] = np.array(im_bone)
            X[i,:,:,1] = np.array(im_brain)

            # Load segmentation mask if training
            if load_Y:
                seg_path = self.data_dir + f"/{pt}/brain/{sl}_HGE_Seg.jpg"

                #Load image mask if it exists
                if os.path.exists(seg_path):
                    seg = Image.open(seg_path)
                    Y[i,:,:] = np.array(seg)

                # Apply augmentation
                if self.augmentation != None:
                    X, Y = self.augmentation(X, Y)

        # Return X and Y if training
        if load_Y:        
            return X, Y
        else:
            # Return X if not training
            return X


#%%

trainLoader = DataLoader(IDs,to_fit=True)
i = 10
X = trainLoader[0]

#for i in range(32):
#    plt.figure()
#    plt.imshow(X[i,:,:,0])


def preprocess(pt,sl):
    """
    123123
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

    np.save(path1+f"{sl}.npy",im1)
    np.save(path2+f"{sl}.npy",im2)
    np.save(path3+f"{sl}.npy",seg)

def normalize1(im):
    max_val = 255
    X_AUG = im/max_val
    return X_AUG

def normalize2(im):
    pass
#%%

ixs = list(range(49,131))
IDs = []

for i, ix in enumerate(ixs):

    f = f"data/Patients_CT/{ix:03d}/bone"
    n_slices = len(os.listdir(f))
    IDs.extend([f"pt_{ix:03d}_sl_{i}" for i in range(1,n_slices+1)])
#%%

for ID in IDs:
    pt = ID.split("_")[1]
    sl = ID.split("_")[3]
    preprocess(pt,sl)

