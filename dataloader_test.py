
#%%

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

data_dir = "data/Patients_CT"

#%%
#Slices for patient 67 for whom a segmentation image exists
im_ids = [17,18,19,20,21]

#Dataset dimensions
X_dim = 650
Y_dim = 650
n_images = len(im_ids)
n_channels = 2 


#X: Images, Y: Masks

#Dimensions in the datastructure
#   Dimension 0: The different patients
#   Dimension 1+2: 2D Image data
#   Dimension 3: Channels (e.g. RGB channels or in our case, using both the "Bone" og "Brain" CT images)

X = np.zeros((n_images,X_dim,Y_dim,n_channels))
Y = np.zeros((n_images,X_dim,Y_dim))

#Load dataset 
for i, ID in enumerate(im_ids):

    # Load bone and brain slice
    im_bone  = Image.open(data_dir + "/067/bone/{}.jpg".format(ID))
    im_brain = Image.open(data_dir + "/067/brain/{}.jpg".format(ID))

    # Load segmentation mask
    d = data_dir + "/067/brain/{}_HGE_Seg.jpg".format(ID)
    if os.path.exists(d):
        mask = np.array(Image.open(d))
    else:
        mask = np.zeros((X_dim,Y_dim))

    # Save the images
    X[i,:,:,0] = np.array(im_bone)
    X[i,:,:,1] = np.array(im_brain)

    Y[i,:,:] = mask


#Show example
plt.figure()
plt.imshow(X[0,:,:,0])

plt.figure()
plt.imshow(Y[0,:,:])

#%% Dataloader
import tensorflow as tf
from tensorflow.keras.utils import Sequence

#Dataloader class based on TF
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
        if shuffle == True:
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
             # Load bone and brain slice
            im_bone  = Image.open(self.data_dir + "/067/bone/{}.jpg".format(ID))
            im_brain = Image.open(self.data_dir + "/067/brain/{}.jpg".format(ID))

            X[i,:,:,0] = np.array(im_bone)
            X[i,:,:,1] = np.array(im_brain)

            # Load segmentation mask if training
            if load_Y:
                seg_path = data_dir + "/067/brain/{}_HGE_Seg.jpg".format(ID)

                #Load image mask if it exists
                if os.path.exists(seg_path):
                    seg = Image.open(seg_path)
                    Y[i,:,:] = np.array(seg)

                # Apply augmentation
                if self.augmentation != None:
                    X, Y = self.augmentation(X, Y)
                
                # Return X and Y if training
                return X, Y
            
            # Return X if not training
            return X

#%%

