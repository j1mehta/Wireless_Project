
# coding: utf-8

# In[1]:

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
#sets constraints (eg. non-negativity) on network parameters during optimization.
from keras.constraints import maxnorm, MinMaxNorm
from keras.optimizers import SGD
from keras.optimizers import adam
import pandas as pd
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:

path_data = '/Users/jatin/Desktop/WirelessProject/DATASET/'
mos_data = pd.read_csv(path_data + r'mos.txt', header = None)


# In[3]:

#keras.backend.backend()


# In[4]:

'''Let's say you're working with 128x128 pixel RGB images (that's 128x128 pixels with 3 color channels).
When you put such an image into a numpy array you can either store it with a shape of (128, 128, 3) 
or with a shape of (3, 128, 128).
The dimension ordering specifies if the color channel comes first (as with theano / "th") 
or if it comes last (as with tensorflow / "tf").'''

keras.backend.image_dim_ordering()


# In[5]:

#model.summary()


# In[6]:

epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False, clipvalue=0.5)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[7]:

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


# In[8]:

def load_train():
    X_train = []
    y_train = []
    y_train = mos_data
    print('Read train images')
    for file in glob.glob(path_data+"distorted_images/*.bmp"):
        img = get_im(file)
        X_train.append(img)

    return X_train, y_train


# In[9]:

train, target = load_train()


# In[10]:

def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# In[11]:

X_train, X_test, y_train, y_test = split_validation_set(train, target, 0.2)


# In[12]:

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[13]:

test_X = X_test[240:]
test_y = y_test[240:]


# In[14]:

X_test = X_test[:240]
y_test = y_test[:240]


# In[15]:

len(X_test)


# In[16]:

'''SHAPES: 1:rows 2: columns 3: channels | Thus, (0,3,1,2) means (no_of_samples, 3, 32, 32) and 
(0,1,2,3) means (no_of_samples, 32, 32, 3)'''
#X_train = X_train.transpose(0,3,1,2)
#X_test = X_test.transpose(0,3,1,2)
print (X_train.shape)


# In[17]:

#Input shape has to be rows, columns, channel (since image_dim_ordering is tf)
input_shape = (32, 32, 3)


# In[18]:

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
def vgg_preprocess(x):
    x = x - vgg_mean
    return x[:, ::-1]


# In[19]:

X_train = vgg_preprocess(X_train)
X_test = vgg_preprocess(X_test)


# In[20]:

#Check if numpy array have NaN
np.isnan(np.sum(X_train))


# In[21]:

not np.any(X_test)


# In[22]:

def CNN():
    model = Sequential()
    model.add(Conv2D(50, (1, 1), input_shape = input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(800, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    return model


# In[23]:

model = CNN()
model.compile(loss='mse', optimizer=sgd)


# In[24]:

model.summary()


# In[25]:

model.fit(X_train, y_train, batch_size=128, nb_epoch=10, verbose=1, validation_data=(X_test, y_test))


# In[26]:

model.save_weights('tid2008.h5')


# In[27]:

res = model.predict(test_X)


# In[28]:

print (mean_squared_error(res, test_y))

