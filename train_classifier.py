# CLASSIFICATION


import numpy as np
from matplotlib import pyplot as plt
import wandb
from wandb.keras import WandbCallback
import random
from dataloader_test import DataLoaderClassification, IDs, preprocess
from data_split import get_data_split_IDs
from test2 import AUG
import tensorflow as tf
from train_unet import load_train_val_data_classifier

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
#%% Load data

diagnosis = ["Intraventricular","Intraparenchymal","Subarachnoid","Epidural","Subdural","No_Hemorrhage"]
diagnosis_dict = {i:x for x,i in zip(diagnosis,range(len(diagnosis)))}

train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val, X_test, Y_test, X_train_test, Y_train_test = load_train_val_data_classifier()

# Define the per-epoch callbacks
def log_image(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    ix = random.randrange(10)
    X_test = X_val[np.newaxis,ix,]
    Y_test = Y_val[np.newaxis,ix,]   
    test_pred_raw = model.predict(X_test)

    plt.figure(figsize=(10,10))
    plt.subplot(2,5,1)
    plt.imshow(X_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(2,5,2)
    plt.imshow(X_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(2,5,3)
    plt.imshow(X_test[0,:,:,2],cmap="gray")
    plt.axis("off")
    s = "\n".join(["{:.0f}".format(Y_test.squeeze()[i]) +  diagnosis[i] + ": " + "{:.2f}".format(np.around(x,3)*100) + "%" for i,x in enumerate(test_pred_raw.squeeze())])

    plt.suptitle(s)
    plt.subplot(2,5,3)
    
    #plt.tight_layout()
    # Log the confusion matrix as an image summary.

    # Use the model to predict the values from the training dataset.
    test_pred_raw_train = model.predict(X_train_test)    
    plt.subplot(2,5,6)
    plt.imshow(X_train_test[0,:,:,0],cmap="gray")
    plt.axis("off")
    plt.title("Brain")
    plt.subplot(2,5,7)
    plt.imshow(X_train_test[0,:,:,1],cmap="gray")
    plt.axis("off")
    plt.title("Bone")
    plt.subplot(2,5,8)
    plt.imshow(X_train_test[0,:,:,2])
    plt.axis("off")
    plt.title(str("train"))
    plt.tight_layout()
    #plt.show()
    print(s)
    wandb.log({"im": plt})



#VGG 16 with some modifications
model = Sequential()

#Input shape is 512,512,2 instead of 224,224,3
model.add(Conv2D(input_shape=(512,512,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=6, activation="sigmoid"))

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

wandb_key = "a631abc6ca522c1ebcef41520d0759e0841b8bed"
wandb.login(key=wandb_key)
wandb.init(project='vgg', entity='keras_krigere')
from tensorflow import keras
image_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_image)
 
#model.fit(batch_size=100,validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
model.fit(d_train, batch_size = 2, epochs=50, callbacks = [image_callback, WandbCallback()], validation_data=(X_val,Y_val), validation_batch_size = 2)