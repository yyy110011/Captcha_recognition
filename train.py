import pandas as pd

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils  import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

from func import load_data

#creat CNN model
print('Creating CNN model...')
tensor_in = Input((75, 100, 3))
tensor_out = tensor_in
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Dropout(0.2)(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Dropout(0.2)(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Dropout(0.2)(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Dropout(0.2)(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)

tensor_out = Flatten()(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)

tensor_out = [Dense(10, name='digit1', activation='softmax')(tensor_out),\
              Dense(10, name='digit2', activation='softmax')(tensor_out),\
              Dense(10, name='digit3', activation='softmax')(tensor_out),\
              Dense(10, name='digit4', activation='softmax')(tensor_out)]


model = Model(inputs=tensor_in, outputs=tensor_out)
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.summary()

x_train, y_train, x_val, y_val = load_data('label.txt', split_threshold=800)
'''
data = pd.read_csv('label.txt', header=None)
di = dict()
for index, row in data.iterrows():
    if (len(str(row[1])) < 4 ):
        row[1] = ('0000' + str(row[1]))[-4:]
    di[row[0]] = str(row[1])
#print(di)
split_th = 800

x_train = []
y_train = []
yListData = [[] for _ in range(4)]
yListVal = [[] for _ in range(4)]
for data_idx, key in enumerate(di.keys()):
    if data_idx > 899:
        break
    img = Image.open(key).convert('RGB')
    x_train.append(np.array(img)/255)
    label = []
    for index, digit in enumerate(di[key]):
        tmp = np.zeros(10)
        tmp[int(digit)] = 1
        label.append(tmp)
        if data_idx > split_th -1:
            yListVal[index].append(tmp)
        
        else:
            yListData[index].append(tmp)
    #y_train.append(label)
x_train = np.array(x_train)

y_train = yListData
y_val = yListVal
#y_train = (y_train)


print(np.shape(x_train))
print(np.shape(y_train))
print(type(y_train))
X_data = x_train
x_val = x_train[split_th:, :, :, :]

x_train = x_train[:split_th, :, :, :]
print(np.shape(x_train))
print(np.shape(x_val))
'''

##

train_history = model.fit(x_train, y_train, batch_size=128, epochs=50000, verbose=1, validation_data=(x_val, y_val))#, callbacks=callbacks_list)

###

#img = Image.open(key).convert('RGB')
x_pred = []
x_pred.append(np.array(test_img)/255)
#x_pred.append(np.array(img)/255)
print(np.shape(x_pred))
pred = model.predict(np.array(x_pred))
ok = 0
for data_idx, key in enumerate(di.keys()):    
    if (data_idx > 899):        
        x_pred = []
        test_img = Image.open(key).convert('RGB')
        x_pred.append(np.array(test_img)/255)
        y_val = str(di[key])
        
        pred = model.predict(np.array(x_pred))
        predVal = ""
        for predC in np.argmax(pred, axis=2):
            predVal = predVal + str(predC[0])
        
        if (predVal == y_val):
            ok=ok+1
        else:
            print(key + ' ' + predVal + ' ' + y_val )
        #pred = model.predict(np.array(x_pred))
        #y_val = [  ]
        #if(pred==y_val):
#            ok=ok+1
ok


