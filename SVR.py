import countNoOfPacketsPerFrame as ppframe

import numpy as np
from sklearn.svm import SVR
from sklearn import model_selection
import pickle
#%%
score_file = '/Users/jatin/Desktop/WirelessProject/CODES/mu500overlapScore.txt'
#%%
pktLost_bucket_perFrame = ppframe.frame_packet_trend[1:]
frame_type = ppframe.frame_type[1:]
loss_bucket=[]
for each in pktLost_bucket_perFrame:
    loss_bucket.append(each[0])
loss_pFrame = ppframe.packet_lost_per_frame[1:]

#%%
zipped_input = list(zip(frame_type,loss_pFrame))

#%%
vec_type_noLoss = np.asarray(zipped_input)
vect_bucket = np.asarray(loss_bucket)
p = [0,0,0,0,0]
for i in range(len(loss_bucket)):
    if(len(loss_bucket[i])==0):
        loss_bucket[i].extend(p)
    loss_bucket[i].extend(zipped_input[i])

with open(score_file) as s:
    y_f = s.readlines()

y = [float(x.strip()) for x in y_f]
#%%
X = loss_bucket[:85400]
Y = y[:85400]
model = SVR()
model1 = SVR()
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
#%%

#model1.fit(X, Y)
### save the model to disk
#filename = 'finalized_model.sav'
#pickle.dump(model1, open(filename, 'wb'))
## 
### some time later...
## 
### load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))
#res = loaded_model.score(X_test, Y_test)
