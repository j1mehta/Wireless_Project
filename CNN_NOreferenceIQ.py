
# coding: utf-8

# In[380]:

import os
import glob
import dis
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
import pandas as pd
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.cross_validation import train_test_split



# In[381]:

path_data = '/Users/jatin/Desktop/WirelessProject/DATASET/'
mos_data = pd.read_csv(path_data + r'mos.txt', header = None)


# In[384]:

keras.backend.backend()


# In[387]:

'''Let's say you're working with 128x128 pixel RGB images (that's 128x128 pixels with 3 color channels).
When you put such an image into a numpy array you can either store it with a shape of (128, 128, 3) 
or with a shape of (3, 128, 128).
The dimension ordering specifies if the color channel comes first (as with theano / "th") 
or if it comes last (as with tensorflow / "tf").'''

keras.backend.image_dim_ordering()


# In[388]:

def CNN():
    model = Sequential()
    #Input shape has to be rows, columns, channel (since image_dim_ordering is tf)
    model.add(Conv2D(50, (7, 7), input_shape=(32, 32, 3),strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(800, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal'))
    return model


# In[405]:

model.summary()


# In[390]:

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)


# In[412]:

model.compile(loss='mse', optimizer="sgd")


# In[413]:

def get_im(path):
    '''loads image as PIL
    Returns
        A 3D Numpy array
    Numpy array x has format (height, width, channel)
    '''
    img = load_img(path)
    size = (32,32)
    resized = img.resize(size) 
    
    return img_to_array(resized)


# In[445]:

def load_train():
    X_train = []
    y_train = []
    y_train = mos_data
    print('Read train images')
    for file in glob.glob(path_data+"distorted_images/*.bmp"):
        img = get_im(file)
        X_train.append(img)

    return X_train, y_train


# In[446]:

train, target = load_train()


# In[447]:

len(y_train)


# In[448]:

def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# In[449]:

X_train, X_test, y_train, y_test = split_validation_set(train, target, 0.2)


# In[450]:

len(X_test)


# In[451]:

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[452]:

X_train.shape


# In[453]:

'''SHAPES: 1:rows 2: columns 3: channels | Thus, (0,3,1,2) means (no_of_samples, 3, 32, 32) and 
(0,1,2,3) means (no_of_samples, 32, 32, 3)'''
X_train = X_train.transpose(0,3,1,2)
X_test = X_test.transpose(0,3,1,2)
print (X_train.shape)


# In[454]:

X_train.shape


# In[ ]:

model.fit(X_train, y_train, batch_size=3, nb_epoch=3, verbose=1, validation_data=(X_test, y_test))


# In[ ]:



