# CLASSIFICATION

#%%
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import scipy
from scipy import ndimage as nd
import csv
from dataloader_test import DataLoader, IDs, preprocess
from data_split import get_data_split_IDs

#%%
hem_data = []

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
#%% split data

# does not work
train_IDs, val_IDs, test_IDs = get_data_split_IDs(IDs)
d_train = DataLoader(train_IDs,batch_size=2)

train_images =1
train_labels = 1

test_images = 1
test_labels =1

#%% Augmentation
# X_AUG, Y_AUG = AUG(X_slice,Y_slice,angles,scales)
# Network


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(512, 512)),
    tf.keras.layers.Dense(43000, activation='relu'),
    tf.keras.layers.Dense(cols-2, activation='sigmoid') # dimension is now equal to number of classes (hemorrhage types)
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)



# dont use softmax
# probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
                                 
predictions = probability_model.predict(test_images)



