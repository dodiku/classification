'''
Machine Learning for Cities // New York University
Instructor: Professor Luis Gustavo Nonato

Written by: Viola Zhong & Dror Ayalon
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn import svm
import matplotlib.pyplot as plt
import csv
import pandas as pd

'''------------------------------------
LOADING THE SPLITED DATA SETS AND TURNING THEM TO NUMPY ARRAYS
------------------------------------'''
df = pd.read_csv('clean_datasets/x_training.csv', delimiter=';', index_col=False)
x_training = df.as_matrix()

df = pd.read_csv('clean_datasets/y_training.csv', delimiter=';', index_col=False)
y_training = df.as_matrix()
y_training = y_training[:,0]

df = pd.read_csv('clean_datasets/x_test.csv', delimiter=';', index_col=False)
x_test = df.as_matrix()

df = pd.read_csv('clean_datasets/y_test.csv', delimiter=';', index_col=False)
y_test = df.as_matrix()
y_test = y_test[:,0]

print ('x_training:')
print (x_training.shape)
print ('y_training:')
print (y_training.shape)

print ('x_test:')
print (x_test.shape)
print (x_test)
print ('y_test:')
print (y_test.shape)



'''------------------------------------
Naive Bayes
------------------------------------'''
gnb = naive_bayes.GaussianNB()
gnb.fit(x_training,y_training)
prediction_gnb = gnb.predict(x_test)

ts, = y_test.shape
error_gnb = np.sum((prediction_gnb[i] != y_test[i]) for i in range(0,ts))
print("----------Naive Bayes----------")
print(error_gnb, "misclassified data out of", ts, "(",error_gnb/ts,"%)")



'''------------------------------------
SVM Linear
------------------------------------'''
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(x_training,y_training)
prediction_svm_linear = svm_linear.predict(x_test)

error_svm_linear = np.sum((prediction_svm_linear[i] != y_test[i]) for i in range(0,ts))
print("----------SVM Linear----------")
print(error_svm_linear, "misclassified data out of", ts, "(",error_svm_linear/ts,"%)")



'''------------------------------------
SVM Kernel
------------------------------------'''
svm_rbf = svm.SVC(kernel='rbf', gamma=500)
# svm_rbf = svm.SVC(kernel='rbf', gamma=100)
svm_rbf.fit(x_training,y_training)
prediction_svm_rbf = svm_rbf.predict(x_test)
error_svm_rbf = np.sum((prediction_svm_rbf[i] != y_test[i]) for i in range(0,ts))
print("----------SVM RBF----------")
print("number of support vectors",len(svm_rbf.support_))
print(error_svm_rbf, "misclassified data out of", ts, "(",error_svm_rbf/ts,"%)")

plt.subplot(1,3,1)
plt.scatter(x_test[:,2],x_test[:,0],c=prediction_gnb, s=10)
plt.title('Naive Bayes:\n%s%% Error' %(int(error_gnb/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1,3,2)
plt.scatter(x_test[:,2],x_test[:,0],c=prediction_svm_linear, s=10)
plt.title('SVM Linear:\n%s%% Error' %(int(error_svm_linear/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(1,3,3)
plt.scatter(x_test[:,2],x_test[:,0],c=prediction_svm_rbf, s=10)
plt.title('SVM Kernel:\n%s%% Error' %(int(error_svm_rbf/ts*100)), fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.show()
