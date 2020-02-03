#Python Program for clustering 2D data. Created by Kerr Fitzgerald 2020.
#Program is based on algorithm 20.2 from "Information Theory, Inference, and Learning Algorithms" by David J.C. MacKay
#The program implements the hard (often known as Naive) K-means algorithm but adds features to jump out low cluster assignment numbers (e.g. min_assign variable).
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

# READ IN DATAFRAME FROM 2 FILES. FILE SHOULD BE CSV WITH ONE FILE CONTAINING COLUMN NAMES.
#col_names=["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]  
col_names =["test_X", "test_Y"]
kmeans_df = pd.read_csv("test_2D.csv", names=col_names)
print('*************************************DATA FRAME***********************************')
print(kmeans_df)

# ASSIGN KEY DATA.
print('**************************************KEY DATA************************************')
x_key = 'test_X'
min_x = (kmeans_df[x_key].min())
max_x = (kmeans_df[x_key].max())
print('The x_key value is: ', x_key)
print('Min x_key value is: ',min_x)
print('Max x_key value is: ',max_x)
y_key = 'test_Y'
min_y = (kmeans_df[y_key].min())
max_y = (kmeans_df[y_key].max())
print('The y_key value is: ', y_key)
print('Min y_key value is: ',min_y)
print('Max y_key value is: ',max_y)

# Assign number of k points (i.e. clusters) and randomly define cordinates based on
# maximum x and y key values.
print('***************************************K DATA*************************************')
K_points = 5 # Number of clusters. Maximum of 10 for plotting.
min_assign = 2 # If a cluster only has this many values then the centorid is randomly moved
K_positions = np.zeros((K_points, 2))
for i in range(0,K_points):
        K_positions[i,0] = np.random.uniform(low=min_x, high=max_x)
        K_positions[i,1] = np.random.uniform(low=min_y, high=max_y)
        print('k',i,' guess co-ordinate is: ','(',K_positions[i,0],',',K_positions[i,1],')')
x = kmeans_df.loc[:, x_key]
y = kmeans_df.loc[:, y_key]

print('*************************************PLOT K-MEANS INITIAL***********************************')
plt.scatter(x,y)
x_k = K_positions[:,0]
y_k = K_positions[:,1]
plt.scatter(x,y)
plt.scatter(x_k, y_k, marker= '*')
plt.show()

dist_tracker = np.zeros(K_points)
K_tracker = np.zeros(len(kmeans_df))
centroidsx = np.zeros(K_points)
centroidsy = np.zeros(K_points)
K_assignment = np.zeros(K_points)

print('*************************************Update K-means***********************************')
for n in range(0,1000):
    for i in range(0,len(kmeans_df)):
        for j in range(0,K_points):
            dist_tracker[j] = np.sqrt((x[i]- K_positions[j,0])**2 + (y[i]- K_positions[j,1])**2)
            #print(j,' dist_tracker value = ',dist_tracker[j])
        for k in range(0,K_points):
                if dist_tracker[k] == np.amin(dist_tracker):
                    #print(k,' selected minimum',np.amin(dist_tracker))
                    centroidsx[k] = centroidsx[k] + x[i]
                    centroidsy[k] = centroidsy[k] + y[i]
                    K_tracker[i] = k+1
                    K_assignment[k] = K_assignment[k] + 1
                #print('K_assignment ',k,' =',K_assignment[k])
    for i in range(0,K_points):
        #print('X Centroid ',i,' =',centroidsx[i])
        if K_assignment[i] <= min_assign:
            K_assignment[i] = 1
            centroidsx[i] = np.random.uniform(low=min_x, high=max_x)
            centroidsy[i] = np.random.uniform(low=min_y, high=max_y)
        K_positions[i,0] = centroidsx[i]/float(K_assignment[i])
        K_positions[i,1] = centroidsy[i]/float(K_assignment[i])
    for q in range(0,K_points):
        centroidsx[q] = 0
        centroidsy[q] = 0
        K_assignment[q] = 0
        #K_tracker[q] = 0
    #print(K_assignment)
#print(K_assignment)

print('*************************************PLOT K-MEANS FINAL***********************************')
for i in range(0,len(kmeans_df)):
    if K_tracker[i] == 1.0:
        plt.scatter(x[i], y[i], c='b')
    if K_tracker[i] == 2.0:
        plt.scatter(x[i], y[i], c='r')
    if K_tracker[i] == 3.0:
        plt.scatter(x[i], y[i], c='g')
    if K_tracker[i] == 4.0:
        plt.scatter(x[i], y[i], c='y')
    if K_tracker[i] == 5.0:
        plt.scatter(x[i], y[i], c='orange')
    if K_tracker[i] == 6.0:
        plt.scatter(x[i], y[i], c='blueviolet')
    if K_tracker[i] == 7.0:
        plt.scatter(x[i], y[i], c='magenta')
    if K_tracker[i] == 8.0:
        plt.scatter(x[i], y[i], c='cyan')
    if K_tracker[i] == 9.0:
        plt.scatter(x[i], y[i], c='palegreen')
    if K_tracker[i] == 10.0:
        plt.scatter(x[i], y[i], c='black') 
plt.scatter(x_k, y_k, c='k', marker= '*')
plt.show()