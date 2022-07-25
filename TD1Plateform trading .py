# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 08:35:00 2020

@author: Léa
"""
#TD 
#Classification we look for an Y that is 0 and 1 and Xi belogn to R
 
#we look for the probability that Y=1 = sigmoid
#imbalance bid-ask/quantité bid et quantité ask (lots) imbalance more bid(resp ask) than ask (resp bid)
#p(y=1)= 1/(1+exp(-x0-X1*imbalance))
# asset liquid -> tick lower, imbalance 

import time
import csv
import numpy as np
import sklearn
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.ndimage.interpolation import shift
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#import linearregression

def importCSV(path):
   file = open(path) 
   dataframe = csv.reader(file, delimiter=';')
   x = list(dataframe)
   result = np.array(x)
   return result

def imbalance(data):
    return (data["BidSize"]- data["AskSize"])/(data["BidSize"]+ data["AskSize"])

#bid ask spread
def mid(data):
    return (data["Bid"]+data["Ask"])/2

def deltamid(data):
    deltamid = (data['mid']-shift(data['mid'],1,cval=np.NaN))
    deltamid= np.where(abs(deltamid)<0.0001,0,deltamid)
    i=len(deltamid)-1
    p=0.5
    while(i>=0):
        while(deltamid[i]==0):
            deltamid[i]=p
            i=i-1
        alpha =deltamid[i] 
        deltamid[i]=p
        p=alpha
        i=i-1

    return deltamid

def Youtput(deltamid):
    Y = np.zeros(len(deltamid))
    #print(deltamid[::-1])
    Y = np.where((deltamid<0) , 0 , 1)
    return Y

def imbalanceL2(data):
    return (data["L2BidSize"]- data["L2AskSize"])/(data["L2BidSize"]+ data["L2AskSize"])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
               test Unitaire
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import unittest

class MyTest(unittest.TestCase):
    def datasame(self,data,size):
        self.assertNotEqual(len(data), size)
    def dataNAappear(self,data,size):
        self.assertEqual(np.sum(np.where(np.isnan(data)==True,1,0)), 0)
    def isoutputNull(self,output):
        self.assertNotEqual(output.sum(), 0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Signal Backtest Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def signal(prediction,real,delta):
    prediction_signal = np.where(prediction>delta,1,-1)
    real=np.where(real>delta,1,-1)
    signal_graphe(prediction_signal,real)  
    return prediction_signal

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                        PNL et Backtest
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def PNL(prediction,data):
    tt= prediction*(shift(data['mid'],-1,cval=0)-data['mid'])
    tt[0]=0
    tt = np.where(abs(tt)<0.0001,0,tt)
    tt[len(tt)-2]=0
    tt[len(tt)-1]=0
    print(tt)
    return np.cumsum(tt)

def PNLexecution(prediction,data):
    tt =np.where(shift(prediction,-1,cval=0)!= prediction,np.where(prediction<0,shift(data['Ask'],-1,cval=0),shift(data['Bid'],-1,cval=0)),0)
    tt2 = np.where(shift(prediction,-1,cval=0)!=prediction,np.where(prediction<0,data['Bid'],data['Ask']),0)
    gain=prediction*(tt-tt2)
    gain[len(gain)-1]=0
    return np.cumsum((gain))


def backtest(prediction,real,start):
    prediction = signal(prediction,real,0.50) 
    real = np.where(real>0.5,1,-1)
    tt = pd.crosstab(real,prediction , rownames=['Real'], colnames=['Predicted'])
    print(tt)
    print("Taux reconnaissance ", metrics.accuracy_score(real,prediction))
    print("Taux d'erreur ",1-metrics.accuracy_score(real,prediction))
    pnl =PNL(prediction,data.iloc[start:,:])
    pnl_exec =PNLexecution(prediction,data.iloc[start:,:])
    pnlgraphe(pnl," mid ")
    pnlgraphe(pnl_exec," excecution ")
    #print(pnl)
    print("PNL Mid ", (pnl[len(pnl)-1]))
    print("PNL execution",(pnl_exec[len(pnl_exec)-1]))

def backtestdd(prediction,real,start,delta):
    prediction = signal(prediction,real,delta) 
    real = np.where(real>delta,1,-1)
    tt = pd.crosstab(real,prediction , rownames=['Real'], colnames=['Predicted'])
    print(tt)
    print("Taux reconnaissance ", metrics.accuracy_score(real,prediction))
    print("Taux d'erreur ",1-metrics.accuracy_score(real,prediction))
    pnl =PNL(prediction,data.iloc[start:,:])
    pnl_exec =PNLexecution(prediction,data.iloc[start:,:])
    pnlgraphe(pnl," mid ")
    pnlgraphe(pnl_exec," excecution ")
    #print(pnl)
    print("PNL Mid ", (pnl[len(pnl)-1]))
    print("PNL execution",(pnl_exec[len(pnl_exec)-1]))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Linear Regression Methods
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def signal_graphe(prediction,real):
    real=np.cumsum(real)#np.where(real==0,-1,1))
    prediction=np.cumsum(prediction)#np.where(prediction==0,-1,1))
    prediction = pd.DataFrame(prediction)
    real = pd.DataFrame(real)
    conc = prediction
    conc[1] =real 
    conc = pd.DataFrame(conc)
    plt.plot(conc)
    plt.title("Signal Pedicted vs signal real ")
    plt.xlabel('Date')
    plt.legend(['prediction','real'])
    plt.ylabel('Signal cumulated Up and Down')
    plt.show()

def pnlgraphe(pnl,title):
    plt.plot(pnl)
    plt.title("Pnl " + title)
    plt.xlabel('time')
    plt.legend(['Pnl'])
    plt.ylabel('Cumulative PNL')
    plt.show()
    
    
def Regresson_results(dataset,col,response,start,stop):
    prediction = startReg(dataset,col,response,start,stop)
    #real = dataset.loc[:,response]
    #real=real.iloc[start:]
    #print("MSE  Linear ",metrics.mean_squared_error(prediction,real))
    #print("MEA Linear ",metrics.mean_absolute_error(prediction,real))
    #signal_graphe(prediction,real)
    return prediction


def startReg(dataset,col,response,start,stop):
    prediction=np.zeros(len(dataset))
    for i in np.arange(start,len(dataset),stop):
        X =dataset.loc[:,col]
        X_pred = X.iloc[i:i+stop,:]
        X = X.iloc[i-stop:i-1,:]
        Y = dataset.loc[:,response]
        Y_pred = Y.iloc[i:i+stop]
        Y=Y.iloc[i-stop:i-1]
        #X_pred=np.array(X_pred).reshape(-1,1)
        if i%stop==0 :
            linearreg = LinearRegression()
            modelLinReg = linearreg.fit(X,Y)
        prediction[i:i+stop] = modelLinReg.predict(X_pred)
    prediction = pd.DataFrame(prediction) 
    return prediction.iloc[start:,:]
 
def multipleRegressionLinear(Y,X,Y_pred,X_pred):
    linearreg = LinearRegression()
    modelLinReg = linearreg.fit(X,Y)
    print("Coef ",modelLinReg.coef_)
    print("X0",modelLinReg.intercept_)
    print("Score ",modelLinReg.score(X,Y))
    prediction = modelLinReg.predict(X_pred)
    return prediction

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Log Linear Regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def startLogReg(dataset,col,response,start,stop):
    prediction=np.zeros(len(dataset))
    for i in np.arange(start,len(dataset),stop):
        X =dataset.loc[:,col]
        X_pred = X.iloc[i:i+stop,:]
        X = X.iloc[i-stop:i-1,:]
        Y = dataset.loc[:,response]
        Y_pred = Y.iloc[i:i+stop]
        Y=Y.iloc[i-stop:i-1]
        #X_pred=np.array(X_pred).reshape(-1,1)
        if i%stop==0 :
            #print(i)
            linearreg = LogisticRegression()
            modelLinReg = linearreg.fit(X,Y)
        prediction[i:i+stop] = modelLinReg.predict(X_pred)
    prediction = pd.DataFrame(prediction) 
    return prediction.iloc[start:,:]

def logReg(Y,X,Y_pred,X_pred):
    logreg = LogisticRegression()
    modelLogReg = logreg.fit(X,Y)
    print("coef",modelLogReg.coef_)
    print("intercept",modelLogReg.intercept_)
    print("score",modelLogReg.score(X,Y))
    prediction = modelLogReg.predict(X_pred)
    
    logit_roc_auc = roc_auc_score(Y_pred, logreg.predict(X_pred))
    fpr, tpr, thresholds = roc_curve(Y_pred, logreg.predict_proba(X_pred)[:,1])
    plt.figure()
    #The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

    return prediction

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Project Starts
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#data= importCSV(r"C:/Users/Léa/Desktop/plateforme de trading/VTX.SX5E_quotes_0.csv")
data = pd.read_csv("C:/Users/Léa/Desktop/plateforme de trading/VTX.SX5E_quotes_0.csv",delimiter=";")

data1= pd.read_csv("C:/Users/Léa/Desktop/plateforme de trading/VTX.SX5E_quotes_1.csv",delimiter=";")
data2= pd.read_csv("C:/Users/Léa/Desktop/plateforme de trading/VTX.SX5E_quotes_2.csv",delimiter=";")
data3= pd.read_csv("C:/Users/Léa/Desktop/plateforme de trading/VTX.SX5E_quotes_3.csv",delimiter=";")
data =data.append(data1)
data =data.append(data2)
data =data.append(data3)

                                               
data['imbalance']= imbalance(data)
data['mid'] =  mid(data)
data['deltamid']= deltamid(data)
data['output']= pd.DataFrame(Youtput(data['deltamid']))
data['imbalanceL2']= imbalanceL2(data)
data['X0']= np.zeros(len(data))


print("Delta mid", data['imbalance'])
print("Mid ",data['mid'] )
print("Imbalance",data['deltamid'])
print("Output ",data['output'])


#Plot Imbalence
data['deltamid'].hist()
plt.title("Mid distribution")
plt.figure()
data['imbalance'].hist(bins = np.arange(-1,1,0.05))
plt.title("Imbalance distribution")
plt.figure()
#plot mid


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Linear Regression
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#important
data['imbalance'] = data['imbalance'].fillna(data['imbalance'].mean())

#Regression Simple

percentlearn = int(((len( data['imbalance']))/2))
X =data.loc[:,['imbalance','X0']]
X_pred = X.iloc[percentlearn:]
X = (X.iloc[:percentlearn])
Y = data.loc[:,'output']
Y_pred = Y.iloc[percentlearn:]
Y=Y.iloc[:percentlearn]

test = MyTest()
test.isoutputNull(Y)

prediction = multipleRegressionLinear(Y,X,Y_pred,X_pred)

#si on fait varier le taux d'apprentissage, on remarque que l'accuracy ne change pas tant que ça
percent =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range(len(percent)):
    percentlearn = int(percent[i]*len( data['imbalance']))
    X =data.loc[:,['imbalance','X0']]
    X_pred = X.iloc[percentlearn:]
    X = (X.iloc[:percentlearn])
    Y = data.loc[:,'output']
    Y_pred = Y.iloc[percentlearn:]
    Y=Y.iloc[:percentlearn]    
    predictionl = multipleRegressionLinear(Y,X,Y_pred,X_pred)
    backtestdd(predictionl,Y_pred,percentlearn,0.5)

start =1000
stop = 1000




#Log regression avec Limite 2
data['imbalance'] = data['imbalance'].fillna(data['imbalance'].mean())
percentlearn = int(((len(data['imbalance']))/2))
X =data.loc[:,['imbalance','imbalanceL2']]
X_pred = X.iloc[percentlearn:]
X = (X.iloc[:percentlearn])
Y = data.loc[:,'output']
Y_pred = Y.iloc[percentlearn:]
Y=Y.iloc[:percentlearn]

prediction222 = multipleRegressionLinear(Y,X,Y_pred,X_pred)




#regression avec réapprentissage
Y = data.loc[:,'output']
Y_pred2 = Y.iloc[stop:]
tic = time.perf_counter()
col =['imbalance','X0']
response ='output'
prediction2= Regresson_results(data,col,response,start,stop)
toc = time.perf_counter()
print("Time :",str(toc-tic))

#regression avec réapprentissage
data['imbalanceL2'] = data['imbalanceL2'].fillna(data['imbalanceL2'].mean())
Y = data.loc[:,'output']
Y_pred2 = Y.iloc[stop:]


tic = time.perf_counter()
col =['imbalance','imbalanceL2']
response ='output'
start = 1000
stop = 1000
prediction22 = Regresson_results(data,col,response,start,stop)
toc = time.perf_counter()
print("Time :",str(toc-tic))



test.dataNAappear(prediction,len(prediction))
test.dataNAappear(prediction2,len(prediction2))
test.dataNAappear(prediction22,len(prediction22))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Backtest Linear Reg
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#reg simple taux d'apprentissage = 50%
backtest(prediction,Y_pred,percentlearn)
#reg avec réapprentissage
backtest(prediction2.iloc[:,0],Y_pred2,stop)
#reg lineaire ordre 2 avec réapprentissage
backtest(prediction22.iloc[:,0],Y_pred2,stop)
#reg simple ordre 2
backtest(prediction222.iloc[:,0],Y_pred2,stop)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Log Linear
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#important
data['imbalance'] = data['imbalance'].fillna(data['imbalance'].mean())
percentlearn = int(((len(data['imbalance']))/2))
X =data.loc[:,['imbalance','X0']]
X_pred = X.iloc[percentlearn:]
X = (X.iloc[:percentlearn])
Y = data.loc[:,'output']
Y_pred = Y.iloc[percentlearn:]
Y=Y.iloc[:percentlearn]

predictionLog = logReg(Y,X,Y_pred,X_pred)


#si on fait varier le taux d'apprentissage, on remarque que l'accuracy ne change pas tant que ça
percent =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range(len(percent)):
    percentlearn = int(percent[i]*len( data['imbalance']))
    X =data.loc[:,['imbalance','X0']]
    X_pred = X.iloc[percentlearn:]
    X = (X.iloc[:percentlearn])
    Y = data.loc[:,'output']
    Y_pred = Y.iloc[percentlearn:]
    Y=Y.iloc[:percentlearn]    
    predictionl = logReg(Y,X,Y_pred,X_pred)
    backtestdd(predictionl,Y_pred,percentlearn,0.5)


#Log regression avec Limite 2
data['imbalance'] = data['imbalance'].fillna(data['imbalance'].mean())
percentlearn = int(((len(data['imbalance']))/2))
X =data.loc[:,['imbalance','imbalanceL2']]
X_pred = X.iloc[percentlearn:]
X = (X.iloc[:percentlearn])
Y = data.loc[:,'output']
Y_pred = Y.iloc[percentlearn:]
Y=Y.iloc[:percentlearn]
predictionLog223 = logReg(Y,X,Y_pred,X_pred)



#regression avec réapprentissage
col =['imbalance','X0']
response ='output'
start =5000
stop = 5000
Y = data.loc[:,'output']
Y_pred2 = Y.iloc[stop:]
predictionLog2= startLogReg(data,col,response,start,stop)


#regression avec réapprentissage
col =['imbalance','imbalanceL2']
response ='output'
start =5000
stop = 5000
predictionLog22 = startLogReg(data,col,response,start,stop)


test.dataNAappear(predictionLog,len(predictionLog))
test.dataNAappear(predictionLog2,len(predictionLog2))
test.dataNAappear(predictionLog22,len(predictionLog22))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Backtest Linear Reg
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#log simple taux apprentissage = 50%
backtest(predictionLog,Y_pred,percentlearn)
#log réapprentissage
backtest(predictionLog2.iloc[:,0],Y_pred2,stop)
#log réapprentissage lim ordre 2
backtest(predictionLog22.iloc[:,0],Y_pred2,stop)
#log simple lim 2
backtest(pd.DataFrame(predictionLog223).iloc[:,0],Y_pred,percentlearn)

       
        
