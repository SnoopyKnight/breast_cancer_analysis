# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:59:04 2017

@author: Cindy
"""
import pandas as pd
import numpy as np
from pandas import DataFrame as df
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from operator import itemgetter


#print("step 1: get titanic data")
#print("================================")
data=pd.read_csv("breast-cancer-wisconsin.data (2).csv")
#print(data.head(10))

#print("step 2: check Breast Cancer data, missing data")
#print("================================")
#print(data.info())
#print(data.describe())
#Missing attribute values: 16
#print(pd.isnull(data.Bare_Nuclei).value_counts())
mat = data.ix[:,1:10]
correlation = mat.corr()
#print(correlation)   #看相關性
correMatrix = df(correlation)
#plt.pcolor(correMatrix)
#plt.show()

#print("step 3: take no missing value rows, feature scaling, normalization")
#print("================================")
#取沒有missing value 的列
data = data[np.isfinite(data['Bare_Nuclei'])]
#x = df.copy(data)
X = data.values
Y = data.loc[:,'Class'].values
#print(data.values)
            
print("step 4: feature scaling, normalization,  model Classification ==>Support Vector Machine")
#====SVM setting
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#tuned_parameters = [{'C': [1]}]

clf = GridSearchCV(SVC(C=1.0),tuned_parameters,cv=5)

clf = SVC()
#print(clf)
total = 0
accuracy = []
for i in range(10):
    X_train,X_test,Y_train,Y_test = tts(X, Y, test_size=0.5, random_state=i)
    
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
    #print(X_train_std)

    clf.fit(X_train_std,Y_train)
    y_predict = clf.predict(X_test_std)
    total = total + accuracy_score(Y_test,y_predict)
    accuracy.append(accuracy_score(Y_test,y_predict)*100)
    
print('avg_accuracy_score = {}'.format(total/10))
accuracy.append(total/10*100)
    #print("Y_test=",Y_test)
    #print("y_predict=",y_predict)

print("================================")
print("step 5: check report")
print("================================")
from sklearn.metrics import classification_report as clf_report
report = clf_report(Y_test,y_predict,digits=5)
print(report)



#====select the important attribute
dic = {}
for i in range(1,10):
    sub = data.ix[:,[i,10]]
    good = sub[sub.Class==2]
    bad = sub[sub.Class==4]
    abss = (good.ix[:,0].mean()-bad.ix[:,0].mean())/((good.ix[:,0].std()+bad.ix[:,0].std())/2)
    dic[sub.columns[0]] = abss
#print(sorted(dic.values()))
newDic = sorted(dic.items(), key=itemgetter(1))
#print(type(newDic))
top = df(newDic, columns=['Attr', 'score'])
print(top.ix[0:4,0])
X = data.loc[:, top.ix[0:2,0]].values
Y = data.loc[:,'Class'].values
print('')            
print("step 4.2: feature scaling, normalization,  model Classification ==>Support Vector Machine")
total = 0
top5_accuracy= []
for i in range(10):
    X_train,X_test,Y_train,Y_test = tts(X, Y, test_size=0.5, random_state=i)
    
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.fit_transform(X_test)
    #print(X_train_std)

    clf.fit(X_train_std,Y_train)
    y_predict = clf.predict(X_test_std)
    print(y_predict)
    total = total + accuracy_score(Y_test,y_predict)
    top5_accuracy.append(accuracy_score(Y_test,y_predict)*100)
    
print('top avg_accuracy_score = {}'.format(total/10))
top5_accuracy.append(total/10*100)

print("================================")
print("step 5.2: check report")
print("================================")
from sklearn.metrics import classification_report as clf_report
report = clf_report(Y_test,y_predict,digits=5)
print(report)
