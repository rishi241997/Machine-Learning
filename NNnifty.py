import os
os.environ['KERAS_BACKEND'] = 'theano'
import keras
import time

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import accuracy
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
import random
random.seed(200)

data=pd.read_csv('intern2.csv', index_col=None,na_values=['NA'])

#date to datetime
# print(data.columns)
# data['Date'] = pd.to_datetime(data['Date'])
#datetime to number
# data['Date'] = pd.to_numeric(data['Date'])
# d={"Date":2018-06-05,"Prev Close":1189.95,"Open Price":1221.10,"High Price":1243.00,"Low Price":1212.00,"Last Price":1241.00,"Close Price":1239.40,"Average Price":1233.14,"Total Traded Quantity":2135078,"Turnover":2.632848e+09,"No. of Trades":69180,"Deliverable Qty":1356014,"% Dly Qt to Traded Qty":63.5}


# data=data.dropna()

#print(data)



# col=['Prev Close','High Price','Low Price','Last Price','Close Price','Average Price']
# x=data_1[col]
# y=data_1['Open Price']
# print(x)

#how many data we will use 
# (should not be more than dataset length )
'''
data_to_use= 50
 
# number of training data
# should be less than data_to_use
train_end =60
 
total_data=len(data)
print(total_data)
#most recent data is in the end 
#so need offset
start=total_data - data_to_use
print(start)
'''
#start=0
total_data=len(data)
#currently doing prediction only for 1 step ahead
# steps_to_predict =1
 
a = data.iloc [:total_data ,0]    #a
b = data.iloc [:total_data ,1]   #b
c = data.iloc [:total_data ,2]   #c
d = data.iloc [:total_data ,3]    # d
e = data.iloc [:total_data ,4]    #e
f = data.iloc [:total_data ,5]    #f
g = data.iloc [:total_data ,6]     #g
h = data.iloc [:total_data ,7]      #h
I = data.iloc [:total_data ,8]      #I
J = data.iloc [:total_data ,9]      #J
K = data.iloc [:total_data ,10]      #K
L = data.iloc [:total_data ,11]      #L
M = data.iloc [:total_data ,12]      #M
N = data.iloc [:total_data ,13]      #N
O = data.iloc [:total_data ,14]      #O
P = data.iloc [:total_data ,15]      #P
Q = data.iloc [:total_data ,16]      #Q
R = data.iloc [:total_data ,17]      #R
S = data.iloc [:total_data ,18]      #S
T = data.iloc [:total_data ,19]      #T
U = data.iloc [:total_data ,20]
'''#U
V = data.iloc [:total_data ,21]      #V
W = data.iloc [:total_data ,22]      #W
X = data.iloc [:total_data ,23]      #X
Y = data.iloc [:total_data ,24]      #Y
Z = data.iloc [:total_data ,25]      #Z
AA = data.iloc [:total_data ,26]      #AA
'''
yt1 = data.iloc [:total_data ,5]    #Close price
    
# print ("yt1 head :")
# print (yt1.head())
# print(a)

# yt1_ = yt1.shift (-1)
yt1_ = yt1.shift (-1)
# print(yt1_)
# data_1 = pd.concat ([yt, vt, yt1,yt1_, yt2, yt3], axis =1)
# data_1. columns = ['yt', 'vt', 'yt1', 'yt1_', 'yt2', 'yt3']
data_1 = pd.concat ([a,b, c,d, e,f,g,h,I,J,K,L,M,N,O,P,Q,R,S,T,U,yt1,yt1_], axis =1)
data_1. columns = ['a','b','c', 'd', 'e','f','g','h','I','J','K','L','M','N','O','P','Q','R','S','T','U','yt1','yt1_']

data_1 = data_1.dropna()
     
#print (data_1)

# # target variable - closed price
# # after shifting
y = data_1 ['yt1_']
#print(y)
 
        
# #       closed,  volume,   open,  high,   low    
# cols =['yt',    'vt',  'yt1', 'yt2', 'yt3']
cols =['a','b','c', 'd', 'e','f','g','h','I','J','K','L','M','N','O','P','Q','R','S','T','U','yt1']
x = data_1 [cols]
 
#print(x)
#print(x)
#print(cols)



def scaling(x,y):
    print("yha tak chal rha")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,shuffle=False)
    #print(x_test)


    # Sizes of dataset, train_ds, test_ds
    dataset_sz = x.shape[0]
    #print(dataset_sz)
    train_sz = x_train.shape[0]
    test_sz = x_test.shape[0]
    #print(test_sz)

    print("yha tak bhi chal gya")
    # reshape our data into 3 dimensions, [batch_size, timesteps, input_dim]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],-1))
    y_train = np.reshape(y_train, (train_sz,-1))
    #print(x_train)
    print(y_train)
    return x_train,x_test,y_train,y_test 



def preprocessing1(cols,x,y):
    
    scaler_x=preprocessing.MinMaxScaler(feature_range=(0,1))
    #print(scaler_x)
    x=np.array(x).reshape((len(x),len(cols)))
    #print(x)
    
    x=scaler_x.fit_transform(x)
    #print(x)

    scaler_y=preprocessing.MinMaxScaler(feature_range=(0,1))
    y=np.array(y).reshape((len(y),1))
    y=scaler_y.fit_transform(y)
    x_train,x_test,y_train,y_test = scaling(x,y)
    return x_train,x_test,y_train,y_test





def modalbuild():
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    #model.add(LSTM(input_dim=len(cols),output_dim=20,return_sequences=True))
    #model.add(Dropout(0.3))

    model.add(LSTM(100,activation = 'tanh', inner_activation = 'hard_sigmoid',return_sequences=False,input_shape =(len(cols), 1)))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=1,activation = 'linear'))

    start = time.time()
    model.compile(loss="mse", optimizer="Adam", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model
    """
    model=Sequential()
    model.add(LSTM(100,activation = 'tanh', inner_activation = 'hard_sigmoid', input_shape =(len(cols), 1) ))
    # model.add(Flatten())
    model.add(Dense(27))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim=1,activation='linear'))
    # model.add(Flatten())
    model.compile(optimizer = 'RMSprop', loss = 'mean_squared_error')
    return model
    """


x_train,x_test,y_train,y_test = preprocessing1(cols,x,y)
print(x_train,x_test,y_train,y_test)

model = modalbuild()
history=model.fit(x_train,y_train,batch_size=768,epochs=5,validation_split=0.1)

#history=model.fit(x_train, y_train, batch_size = 25, epochs = 50,shuffle=False)

#print(history)


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],-1))

score_train = model.evaluate (x_train, y_train, batch_size =1)
score_test = model.evaluate (x_test, y_test, batch_size =1)
print (" in train MSE = ", round( score_train ,4)) 
print (" in test MSE = ", score_test )
print(100-round( score_train ,4))
print(100-score_test)

# print(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],-1))
pred1 = model.predict (x_test) 
pred1 = scaler_y.inverse_transform (np. array (pred1). reshape ((len( pred1), 1)))


#print(pred1)
prediction_data = pred1[-1]



print ("Inputs: {}".format(model.input_shape))
print ("Outputs: {}".format(model.output_shape))
print ("Actual input: {}".format(x_test.shape))
print ("Actual output: {}".format(y_test.shape))



print ("prediction data:")
print (prediction_data)



print ("actual data")
x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test), len(cols))))
#print (x_test)



plt.plot(pred1, label="predictions",c='g')
#print(pred1)



y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))
plt.plot( [row[0] for row in y_test], label="actual",c='r')
print(y_test)
print(pred1)

xy=[x_test,y_test,pred1]
print(xy)

plt.figure(figsize=(10,5))
plt.plot(pred1,label="predicted",c='g')
plt.plot([row[0] for row in y_test],label="actual",c='r')
plt.legend()
plt.show()


#print(x_test[-1])
#print(pred1[-1])
