'''
Machine Learning for Cities // New York University
Instructor: Professor Luis Gustavo Nonato

Written by: Viola Zhong & Dror Ayalon
'''

import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv


# GrossSqFt, GrossIncomeSqFt, MarketValueperSqFt
# Classify “Neighborhood”

'''
LOADING THE DATASET THAT INCLUDES THE FOLLOWING COLUMNS:
[0] = Neighborhood [classes]
[1] = BldClassif
[2] = YearBuilt
[3] = GrossSqFt [independent]
[4] = GrossIncomeSqFt [independent]
[5] = MarketValueperSqFt [independent]
'''

dataset = open('manhattan-dof.csv', "r")
dataset = csv.reader(dataset,delimiter=';')
next(dataset) # moving passed the title row

'''
CREATING A FLOAT-BASED NUMPY ARRAY
'''
np_array = np.array([r for r in dataset])
np_array = np_array.astype(np.float) # this numpy array holds all the raw data

filtered_data = np.copy(np_array) # this numpy array will hold the data after filterring
print ('##### filtered_data size before cleaning: ', filtered_data.shape)

'''
FILTERING THE DATA (to get rid of outliers + irrelevant data)
'''

removed_rows = 0

# -----------------------------
# cleaning [0] = Neighborhood
# -----------------------------
# removing all relevant rows for neighborhoods bigger than 30

outlier = np.where(filtered_data[:,0] > 30)
print ("[0] = Neighborhood outlier:")
print (outlier)

for i in outlier[0]:
    outlier_filtered = np.where(filtered_data[:,0] > 30)
    # print ("[0] = Neighborhood: outliers left:")
    # print (outlier_filtered)
    # print ('---> [0] = Neighborhood: deleting the following row: ', filtered_data[outlier_filtered[0][0]])
    filtered_data = np.delete(filtered_data, (outlier_filtered[0][0]), axis=0)
    removed_rows = removed_rows + 1

# -----------------------------
# cleaning [3] = GrossSqFt
# -----------------------------
# removing all relevant rows for GrossSqFt bigger than 1,500,000

outlier = np.where((filtered_data[:,3] > 1500000))
print ("[3] = GrossSqFt:")
print (outlier)

# print ('##### filtered_data size before cleaning: ', filtered_data.shape)
for i in outlier[0]:
    outlier_filtered = np.where((filtered_data[:,3] > 1500000))
    # print ("[3] = GrossSqFt: outliers left:")
    # print (outlier_filtered)
    # print ('---> [3] = GrossSqFt: deleting the following row: ', filtered_data[outlier_filtered[0][0]])
    filtered_data = np.delete(filtered_data, (outlier_filtered[0][0]), axis=0)
    removed_rows = removed_rows + 1


# -----------------------------
# cleaning [4] = GrossIncomeSqFt
# -----------------------------
# removing all relevant rows for GrossIncomeSqFt bigger than 50, or smaller than 10

outlier = np.where(((filtered_data[:,4] < 10) | (filtered_data[:,4] > 50)))
# outlier = np.where(filtered_data[:,4] < 10)
print ("[4] = GrossIncomeSqFt:")
print (outlier)

# print ('filtered_data size before cleaning: ', filtered_data.shape)
for i in outlier[0]:
    outlier_filtered = np.where(((filtered_data[:,4] < 10) | (filtered_data[:,4] > 50)))
    # print ("[4] = GrossIncomeSqFt: outliers left:")
    # print (outlier_filtered)
    # print ('---> [4] = GrossIncomeSqFt: deleting the following row: ', filtered_data[outlier_filtered[0][0]])
    filtered_data = np.delete(filtered_data, (outlier_filtered[0][0]), axis=0)
    removed_rows = removed_rows + 1


# -----------------------------
# cleaning [5] = MarketValueperSqFt
# -----------------------------
# removing all relevant rows for MarketValueperSqFt bigger than 50, or smaller than 10

# outlier = np.where((filtered_data[:,5] < 40))
# print ("[5] = MarketValueperSqFt:")
# print (outlier)
#
# for i in outlier[0]:
#     outlier_filtered = np.where((filtered_data[:,5] < 40))
#     print ("[5] = MarketValueperSqFt: outliers left:")
#     print (outlier_filtered)
#     print ('---> [5] = MarketValueperSqFt: deleting the following row: ', filtered_data[outlier_filtered[0][0]])
#     filtered_data = np.delete(filtered_data, (outlier_filtered[0][0]), axis=0)
#     removed_rows = removed_rows + 1


# -----------------------------
# plotting the data after clearning: before vs. after
# -----------------------------

titles = ['Neighborhood', 'BldClassif', 'YearBuilt', 'GrossSqFt', 'GrossIncomeSqFt', 'MarketValueperSqFt']

# generating a plot per column

for i in range(0,6):
    if (i == 0):
        # print (i, 'Neighborhood')
        plt.figure(i)
        plt.plot(np_array[:,2],np_array[:,i], 'o', c="orange", markersize=3)
        plt.plot(filtered_data[:,2],filtered_data[:,i], 'o', c="palegreen", markersize=3)
        plt.title(titles[i], fontsize= 10)
        plt.xlabel('YearBuilt', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        path = "plots/data_cleaning/" + str(i) + ".png"
        plt.savefig(path)
    else:
        plt.figure(i)
        plt.plot(np_array[:,0],np_array[:,i], 'o', c="orange", markersize=3)
        plt.plot(filtered_data[:,0],filtered_data[:,i], 'o', c="palegreen", markersize=3)
        # if (i == 2):
        #     plt.title("Built Year: Original Data (orange) VS. Cleaned Data (green)", fontsize= 10)
        # elif (i == 4):
        #     plt.title("Original Price(SqFt): Original Data (orange) VS. Cleaned Data (green)", fontsize= 10)
        # else:
        #     print (i)
        #     plt.title(titles[i], fontsize= 10)
        plt.title(titles[i], fontsize= 10)
        plt.xlabel('Neighborhood', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        path = "plots/data_cleaning/" + str(i) + ".png"
        plt.savefig(path)

# generating a single plot with all relevant graphs

plt.figure(10)
plt.subplot(2,2,1)
plt.plot(np_array[:,2],np_array[:,0], 'o', c="orange", markersize=3)
plt.plot(filtered_data[:,2],filtered_data[:,0], 'o', c="palegreen", markersize=3)
plt.xlabel('YearBuilt', fontsize=8)
plt.title(titles[0], fontsize= 10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(2,2,2)
plt.plot(np_array[:,0],np_array[:,3], 'o', c="orange", markersize=3)
plt.plot(filtered_data[:,0],filtered_data[:,3], 'o', c="palegreen", markersize=3)
plt.title(titles[3], fontsize= 10)
plt.xlabel('Neighborhood', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(2,2,3)
plt.plot(np_array[:,0],np_array[:,4], 'o', c="orange", markersize=3)
plt.plot(filtered_data[:,0],filtered_data[:,4], 'o', c="palegreen", markersize=3)
plt.title(titles[4], fontsize= 10)
plt.xlabel('Neighborhood', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.subplot(2,2,4)
plt.plot(np_array[:,0],np_array[:,5], 'o', c="orange", markersize=3)
plt.plot(filtered_data[:,0],filtered_data[:,5], 'o', c="palegreen", markersize=3)
plt.title(titles[5], fontsize= 10)
plt.xlabel('Neighborhood', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
path = "plots/data_cleaning/10.png"
plt.savefig(path)

'''
SAVING A NEW CSV FILE WITH THE FILETERED DATA
'''
np.savetxt("clean_datasets/filtered_data.csv", filtered_data, delimiter=';')
print ('#####', removed_rows, 'were removed from the data set')
print ('##### filtered_data size after cleaning: ', filtered_data.shape)
plt.show()
