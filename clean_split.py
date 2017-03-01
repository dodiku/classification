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
** [0] = Neighborhood
[1] = BldClassif
[2] = YearBuilt
* [3] = GrossSqFt
* [4] = GrossIncomeSqFt
* [5] = MarketValueperSqFt
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
outlier = np.where(filtered_data[:,0] > 30) # removing all relevant rows for neighborhoods bigger than 30
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
# outlier = np.where((filtered_data[:,3] > 300000))
# print ("[3] = GrossSqFt:")
# print (outlier)
#
# # print ('##### filtered_data size before cleaning: ', filtered_data.shape)
# for i in outlier[0]:
#     outlier_filtered = np.where((filtered_data[:,3] > 300000))
#     print ("[3] = GrossSqFt: outliers left:")
#     print (outlier_filtered)
#     print ('---> [3] = GrossSqFt: deleting the following row: ', filtered_data[outlier_filtered[0][0]])
#     filtered_data = np.delete(filtered_data, (outlier_filtered[0][0]), axis=0)
# print ('##### filtered_data size after cleaning: ', filtered_data.shape)


# -----------------------------
# cleaning [4] = GrossIncomeSqFt
# -----------------------------
outlier = np.where(((filtered_data[:,4] < 10) | (filtered_data[:,4] > 45)))
# outlier = np.where(filtered_data[:,4] < 10)
print ("[4] = GrossIncomeSqFt:")
print (outlier)

# print ('filtered_data size before cleaning: ', filtered_data.shape)
for i in outlier[0]:
    outlier_filtered = np.where(((filtered_data[:,4] < 10) | (filtered_data[:,4] > 45)))
    print ("[4] = GrossIncomeSqFt: outliers left:")
    print (outlier_filtered)
    print ('---> [4] = GrossIncomeSqFt: deleting the following row: ', filtered_data[outlier_filtered[0][0]])
    filtered_data = np.delete(filtered_data, (outlier_filtered[0][0]), axis=0)


# -----------------------------
# cleaning [5] = MarketValueperSqFt
# -----------------------------
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


'''
SAVING A NEW CSV FILE WITH THE FILETERED DATA
'''
np.savetxt("clean_datasets/filtered_data.csv", filtered_data, delimiter=';')
print ('#####', removed_rows, 'were removed from the data set')
print ('##### filtered_data size after cleaning: ', filtered_data.shape)
# plt.show()
