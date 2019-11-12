#these are code snippets, functions only, each user will need to adjust the code to his/her own needs

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from keras.models import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#getting the data
def proc(t): #t is the name of the excel file
    fi=pd.read_excel(t, index_col=0) #skiprows=3 allows you to start at 3/4
    g=fi.drop(fi.index[0])
    return g

def indx(f): #this is the index DataFrame
    return
#function that keeps only the business days/weekdays
def bsd(a):
    isBusinessDay = BDay().onOffset
    match_series = pd.to_datetime(a.index).map(isBusinessDay) #bC is the DataFrame we work on
    a=a[match_series]
    return a

def fbuild(a, stock):
    #this creates the necessary basic DF to work with in the LSTM
    #stock is the code of the stock to analyze
    #a is the full market basic DF, with last, volume, high, low values
    b=pd.DataFrame(index=a.index) # initiating the new DataFrame that will hodl our datatime
    s=stock+' JT Equity'
    v=s+'.1'
    l=s+'.2'
    h=s+'.3'
    b['p']=a[s]
    b['v']=a[v]
    b['h_l']=(a[h]-a[l])/a[s]
    #initializing the other columns @ 0
    cols=['p-1', 'p-2', 'p-3', 'p-10', 'p-20', 'p-40']
    for k in cols:
        b[k]=0
    for i in range(41, len(b['p'])):
        for ii in cols:
            b[ii].iloc[i]=b['p'].iloc[i]/b['p'].iloc[i-int(ii[2:])]
    c=pd.DataFrame(index=b.index)
    c['tp']=b['p']
    c=c[41:]
    c=c.shift(-1)
    c.iloc[-1]=c.iloc[-2]
    b=b[41:]
    return b, c #c is the target value to forecast. b is the input value. Can always be modified by adding new data

#adding new data from other pandas, that may have different length but same data type indices
#assuming last entry has SAME date index
def adj(b,new):
    #b is the dataFrame from above, c is the new DataFrame
    start=new.index.get_loc(b.index[0])
    new=new[start:]
    ln=list(new)
    for jj in ln:
        b[jj]=new[jj]
    return b


def processData(d,e,lb):
    X,Y = [],[]
    for i in range(len(d)-lb-1):
        X.append(d[i:(i+lb),0])
        Y.append(e[(i+lb),0])
    return np.array(X),np.array(Y)


def frcst(b, c, frame): #using the b,c variables obtained above. More flexible data manipulation, update
    #building the input and output pandas
    sb=MinMaxScaler()
    sc=MinMaxScaler()
    b=sb.fit_transform(b)
    c=sc.fit_transform(c)
    #Creating the training and testing data for LSTM
    X,y=processData(b,c,7)
    X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
    y_train,y_test = y[:int(y.shape[0]*0.80)],y[int(y.shape[0]*0.80):]
    #BUILDING and INITIALIZING the model
    model = Sequential()
    model.add(LSTM(256,input_shape=(7,1)))
    #model.add(LSTM(256,return_sequences = True, input_shape=(7,1))) #shoudl be same as the lb value above
    #model.add(LSTM(256))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    #Reshape data for (Sample,Timestep,Features)
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    history = model.fit(X_train,y_train,epochs=200,validation_data=(X_test,y_test),shuffle=False)
    #testing results
    Xt = model.predict(X_test[-frame:])
    print('last values: ', sc.inverse_transform(Xt))
    plt.plot(sc.inverse_transform(y_test[-frame:].reshape(-1,1)))
    plt.plot(sc.inverse_transform(Xt))
    plt.show()
    return

#running the program
stocks= #the excel file for the stocks. This is a DataFrame, where each stock has last_price, volume, low of the day, high of the day, in this specific order
inx_file= #the excel file for the TPX sector indices
input= #the code of the stock to check
def next_day(stocks, inx_file, input):
    st=proc(stocks)
    st=st.astype(float)
    inx=proc(inx_file)
    inx=inx.drop(inx.index[0])
    inx=inx.astype(float)
    st=bsd(st)
    inx=bsd(inx)
    bb,cc = fbuild(st, input)
    lx=list(inx)
    for i in lx:
        if 'Index' in i:
            bb[i]=inx[i]
    #bb=adj(bb,inx)
    #frame below is how many days to be added last
    frcst(bb, cc, 10) #checking the last 10 days of the testing variable
