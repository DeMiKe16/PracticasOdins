import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.layers import Flatten
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

train_path = os.getcwd()
train_path = train_path + "/Dataset/Cliente1/train/"


train_images = []       
train_labels = []
img_size = 224
shape = (img_size,img_size)  

for filename in os.listdir(train_path):
    if filename.split('.')[1] == 'jpg':
        img = cv2.imread(os.path.join(train_path,filename))
        train_labels.append(filename.split('_')[0])
        img = cv2.resize(img,shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_images.append(img)

train_labels = pd.get_dummies(train_labels).values # one-hot encoding
x_train = np.array(train_images)
y_train = train_labels
x_train,x_val,y_train,y_val = train_test_split(x_train,train_labels,random_state=42)