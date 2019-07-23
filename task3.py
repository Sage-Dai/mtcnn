# -*- coding: utf-8 -*-
"""
@author: 我
"""
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

O_path = './faceImages/faceImageGray/'

def getimgnames(path = O_path):
    filenames = os.listdir(path)
    address = []
    labels = []
    for i in range(10):
        filenames_next = os.listdir(path+'/'+filenames[i])
        for j in range(len(filenames_next)):
            filenames_next[j]= path + '/'+filenames[i] + '/'+filenames_next[j]
            address.append(filenames_next[j])
            labels.append(i)
    return address,labels

img_path,name_labels = getimgnames(path = O_path )

'''
n = len(img_path)
data = np.zeros((n,28,28),dtype='float32') 
'''
n = len(img_path)
shape = (32,32)
data = np.zeros((n,32,32,3),dtype='float32') 
labels = name_labels   #每张图片的标签

for i in range(n):
    img = cv2.imread(img_path[i])
    da_new = cv2.resize(img,shape)
    da_new = da_new[:,:,:]/255          #
    data[i,:,:,:] = da_new
    
data_train,data_test,\
labels_train,labels_test = \
train_test_split(data,labels,test_size = 0.2,random_state = 42)

np.save('./data_train.npy',data_train)
np.save('./data_test.npy',data_test)
np.save('./labels_train.npy',labels_train)
np.save('./labels_test.npy',labels_test)