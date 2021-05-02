# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:24:49 2021

@author: simon
"""
import tensorflow as tf 
from keras import backend as K
import numpy as np 
import matplotlib.pyplot as plt
from dataloader_test import DataLoader, IDs, load_train_val_data


train_IDs, val_IDs, test_IDs, d_train, X_val, Y_val, X_val_test, Y_val_test, X_train_test, Y_train_test, X_test, Y_test = load_train_val_data()


#%%
from test2 import AUG

X = X_test[0,:,:,:]
Y = Y_test[0,:,:]

XX, YY = AUG(X,Y)

plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.imshow(X[:,:,0],cmap='gray')
plt.subplot(2,2,2)
plt.imshow(XX[:,:,0],cmap='gray')

plt.subplot(2,2,3)
plt.imshow(X[:,:,1],cmap='gray')
plt.subplot(2,2,4)
plt.imshow(XX[:,:,1],cmap='gray')

