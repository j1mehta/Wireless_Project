import glob
import scipy.io
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from sklearn.metrics import mean_squared_error as mse
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D
from keras.preprocessing.image import array_to_img, img_to_array, load_img

#%%
#GLOBAL VARIABLES
path_data = '/Users/jatin/Desktop/WirelessProject/jpeg2000 database release/SourceFramesLNormzd/'
#path_data = '/Users/jatin/Desktop/WirelessProject/DATASET/'
#target_path = pd.read_csv(path_data + r'mos.txt', header = None)
#path_test_data = '/Users/jatin/Desktop/WirelessProject/jpeg2000 database release/LNormalizedTest/'
path_test_data = '/Users/jatin/Desktop/WirelessProject/jpeg2000 database release/Few/'
#path_test_data = '/Users/jatin/Desktop/WirelessProject/Out/'
#path_data = '/Users/jatin/Desktop/WirelessProject/jpeg2000 database release/trial/'
target_path = '/Users/jatin/Desktop/WirelessProject/jpeg2000 database release/database release/'

jpeg_sc1 = scipy.io.loadmat(target_path + 'scores1.mat')
jpeg_sc2 = scipy.io.loadmat(target_path + 'scores2.mat')
score = []
for i in range(len(jpeg_sc1['scores'])):
    score.append(jpeg_sc1['scores'][i].mean()*2)
for i in range(len(jpeg_sc2['scores'])):
    score.append(jpeg_sc2['scores'][i].mean()*2)

#input_shape = (60, 60, 3)
input_shape = (30, 30, 3)

epochs = 25
lrate = 0.0001
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, nesterov=True, clipvalue=0.5, decay = decay)
patch_size = 30
nb_patches = 480/patch_size
#%%
def CNN(weights_path=None):
    model = Sequential()
    model.add(Conv2D(50, (7, 7), input_shape = input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(40, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(ZeroPadding2D((1,1)))
    #model.add(Conv2D(40, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(800, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss='mse', optimizer=sgd)
    return model

def get_im(path):
    '''loads image as PIL
    Returns
        A 3D Numpy array
    Numpy array x has format (height, width, channel)
    '''
    img = load_img(path)
    size = (480,480)
    resized = img.resize(size)    
    return img_to_array(resized)

def patchify(img, patch_size, img_score=None):
    '''Reutrns non-overlapping patches
    of input images of size = (patch_size X patch_size)
    and the corresponding score for each patch
    '''
    nb_patch = int((img.size[0]/patch_size))
    patch_array = []
    patch_score_array = []
    ps = patch_size
    rowU = 0
    colL = 0
    rowD = ps
    colR = ps
    img_no = 0
    for i in range(nb_patch):    
        for j in range(nb_patch):
            imgC = img.crop((colL, rowU, colR, rowD))
            patch_array.append(img_to_array(imgC))
            if(img_score):
                patch_score_array.append(img_score)
            colL += ps
            colR += ps
            img_no += 1 
        rowU += ps
        rowD += ps
        colL = 0
        colR = ps
    if(img_score):
        return patch_array, patch_score_array
    else:
        return patch_array

def load_train(path_train, nb_train_img):
    '''load training images'''
    X_train = []
    y_train = score[0:nb_train_img]
    print('Read train images')
    i = 0
    for file in glob.glob(path_train+"img*.bmp"):
        img = get_im(file)
        X_train.append(img)
        if(i==nb_train_img-1):
            break
        i += 1
    return X_train, y_train

def load_test(path_test):
    X_test = []
    print('Read test images')
    for file in glob.glob(path_test+"*.bmp"):
        img = get_im(file)
        X_test.append(img)
    return X_test

def get_train_patches(X_train, y_train): 
    '''Return training patch array and target
    patch array'''
    nb = len(X_train)
    X_train_patch = []
    y_train_patch = []
    for i in range(len(X_train)):
        patch_array, patch_score_array = patchify(array_to_img(X_train[i]),patch_size, y_train[i])
        X_train_patch.append(patch_array)
        y_train_patch.append(patch_score_array)
    X_train_patch = np.array(X_train_patch)
    y_train_patch = np.array(y_train_patch)
    X_train_patch = X_train_patch.reshape(nb_patches*nb_patches*nb, patch_size, patch_size, 3)
    y_train_patch = y_train_patch.reshape(nb_patches*nb_patches*nb,1)
    return X_train_patch, y_train_patch

def get_test_patches(X_test): 
    '''Return test image patch array for prediction'''
    nb = len(X_test)
    X_test_patch = []
    for i in range(len(X_test)):
        patch_array = patchify(array_to_img(X_test[i]),patch_size)
        X_test_patch.append(patch_array)
    X_test_patch = np.array(X_test_patch)
    X_test_patch = X_test_patch.reshape(nb_patches*nb_patches*nb, patch_size, patch_size, 3)
    return X_test_patch

# In[13]:
#X_train, y_train = load_train(path_data, 226)
#X_train_patch, y_train_patch = get_train_patches(X_train, y_train)
#X_train_patch = X_train_patch
#%%
#callbacks = [EarlyStopping(monitor='val_loss', patience=, verbose=0, mode='min'), 
#            ModelCheckpoint('weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', 
#            verbose=0, period=1, save_best_only=True, save_weights_only=True, mode='min')]
#model = CNN()
#model.fit(X_train_patch, y_train_patch, batch_size=32, epochs=20, verbose=1, shuffle = True)
#model.save_weights('/Users/jatin/Desktop/WirelessProject/Weights/jpeg_30_patchSz_Img_20EpochsSHALLOWonMP4.h5')
#%%
model = CNN('/Users/jatin/Desktop/WirelessProject/Weights/jpeg_30_patchSz_Img_20EpochsSHALLOWonMP4.h5')
X_test = load_test(path_test_data) 
X_test_patch = get_test_patches(X_test)
result = (model.predict(X_test_patch))
res = result/256
pred = [sum(res[current: current+256]) for current in range(0, len(res), 256)]
#print(mse(score_orig, pred)) 
    
    
#%%

#def data_generator(path_data, batch_size):
#    X_train_patch = []
#    target_patch = []
#    count = 0
#    while(True):
#        #When no of batches in an epoch are yielded, WHILE LOOP is hit again. This means
#        #fit_generator() destroys the used data in previous epoch
#        #and move on repeating the same process in new epoch.
#        print("While loop starts again")
#        for file in glob.glob(path_data+"img*.bmp"):
#            print("FILE: ", file)
#            print("count: ", count)
#            img = get_im(file)
#            patch_array, patch_score_array = patchify(array_to_img(img),60, score_trial[count])
#            X_train_patch.append(patch_array)
#            target_patch.append(patch_score_array)
#            count += 1
#            if(count % batch_size == 0):
#                print("YIELDING: ")
#                X_train = np.array(X_train_patch)
#                target = np.array(target_patch)
#                print("X_train.shape", X_train.shape)
#                print("target.shape", target.shape)
#                break
#        yield(X_train.reshape(64*count, 60, 60, 3), target.reshape(64*count,1))

