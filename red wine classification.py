# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 22:14:57 2018

@author: aayush
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#importing dataset
dataset = pd.read_csv('winequality-red.csv')
X= dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature Selection 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select = SelectKBest(chi2, k = 5)
select.fit(X,y)
score_features  = select.scores_
X = select.transform(X)

#splitting the data into test set and training set
from sklearn.cross_validation import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y , test_size = 100, random_state = 1 )

#feature scaling
from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
scaleX.fit(X_train)
X_train = scaleX.transform(X_train)
X_test = scaleX.transform(X_test)


#Model
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.svm import SVC
classifier = ovr(SVC(kernel = 'rbf', degree = 4, C=30))
classifier.fit(X_train,y_train)

#Accuracy
from sklearn.metrics import accuracy_score as acc
acc(y_train, classifier.predict(X_train))
acc(y_test, classifier.predict(X_test))

