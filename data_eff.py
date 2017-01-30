# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:23:24 2016

@author: mhurtgen
"""
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn import LinearRegression

from sklearn import linear_model

from sklearn import preprocessing
X=pd.read_csv('ENB2012_data.csv',sep=',')
#print(X.head(3))
#print(X.dtypes)


def g1(row):
    newval=float(row['X1'].replace(',','.'))
    return newval

def g2(row):
    newval=float(row['X2'].replace(',','.'))
    return newval    

def g3(row):
    newval=float(row['X3'].replace(',','.'))
    return newval    

def g4(row):
    newval=float(row['X4'].replace(',','.'))
    return newval    

def g5(row):
    newval=float(row['X5'].replace(',','.'))
    return newval    

def g7(row):
    newval=float(row['X7'].replace(',','.'))
    return newval    

def gy1(row):
    newval=float(row['Y1'].replace(',','.'))
    return newval    

def gy2(row):
    newval=float(row['Y2'].replace(',','.'))
    return newval    



X.X1=X.apply(g1,axis=1)
X.X2=X.apply(g2,axis=1)
X.X3=X.apply(g3,axis=1)
X.X4=X.apply(g4,axis=1)
X.X5=X.apply(g5,axis=1)
X.X7=X.apply(g7,axis=1)

X.Y1=X.apply(gy1,axis=1)
X.Y2=X.apply(gy2,axis=1)

X.X1=pd.to_numeric(X.X1,errors='coerce')
X.X2=pd.to_numeric(X.X2,errors='coerce')
X.X3=pd.to_numeric(X.X3,errors='coerce')
X.X4=pd.to_numeric(X.X4,errors='coerce')
X.X5=pd.to_numeric(X.X5,errors='coerce')
X.X6=pd.to_numeric(X.X6,errors='coerce')
X.X7=pd.to_numeric(X.X7,errors='coerce')
X.X8=pd.to_numeric(X.X8,errors='coerce')

X.Y1=pd.to_numeric(X.Y1,errors='coerce')
X.Y2=pd.to_numeric(X.Y2,errors='coerce')
#print(X)
X=X.dropna(axis=0)
lg=len(X)
#print(X.dtypes)
y=X[['Y1', 'Y2']]
X=X.drop(labels=['Y1','Y2'], axis=1)

X.X6 = X.X6.astype("category").cat.codes
X.X8 = X.X8.astype("category").cat.codes


X = pd.get_dummies(X,columns=['X6','X8'])


print(len(y))
#
#print(y.head(3))
#y = pd.DataFrame(X,columns=[X.Y1,X.Y2])
print(y.head(3))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

#process = preprocessing.Normalizer().fit(X_train)
process = preprocessing.StandardScaler().fit(X_train)
#process = preprocessing.MinMaxScaler().fit(X_train)
#process = preprocessing.MaxAbsScaler().fit(X_train)
#T = preprocessing.MinMaxScaler().fit_transform(df)
#T = preprocessing.MaxAbsScaler().fit_transform(df)
#T = preprocessing.Normalizer().fit_transform(df)

X_train=process.transform(X_train)
X_test=process.transform(X_test)



model=linear_model.LinearRegression()



model.fit(X_train,y_train)

score=model.score(X_test,y_test)

print(score)
print(model.coef_)