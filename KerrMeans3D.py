#PYTHON PROGRAM FOR CLUSTERING 3D DATA. CREATED BY KERR FITZGERALD 2020.
#PROGRAM IS BASED ON ALGORITHM 20.2 FROM "INFORMATION THEORY, INFERENCE, AND LEARNING ALGORITHMS" BY DAVID J.C. MACKAY
#THE PROGRAM IMPLEMENTS THE HARD (I.E NAIVE) k-MEANS ALGORITHM BUT ADDS FEATURES TO JUMP OUT OF LOW CLUSTER ASSIGNMENT NUMBERS (E.G. min_assign VARIABLE).

#IMPORT RELEVANT PYTHON MODULES
import pandas as pd
import sys
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#USER INPUT SELECTED FROM COMMAND LINE
cla = sys.argv
datafile = cla[1]
k_points = int(cla[2])
x_key = cla[3]
y_key = cla[4]
z_key = cla[5]
interest_key =cla[6]
iterations = int(cla[7])
min_assign = int(cla[8])

#SUMMARISE INPUT FOR USER
print("Datafile being used is   : ", datafile)
print("Number of clusters is    : ", k_points)
print("X key is selected as     : ", x_key)
print("Y key is selected as     : ", y_key)
print("Z key is selected as     : ", z_key)
print("Interest key selected is : ", interest_key)

#READ IN DATAFRAME FROM CSV FILE. NOTE THAT FIRST ROW MUST CONTAIN COLUMN HEADERS.    
print('*************************************DATA FRAME***********************************')
kmeans_df = pd.read_csv(datafile, delimiter = ',')
print(kmeans_df)

#SUMMARISE KEY DATA FOR USER
print('**************************************KEY DATA************************************')
min_x = (kmeans_df[x_key].min())
max_x = (kmeans_df[x_key].max())
print('The x_key value is: ', x_key)
print('Min x_key value is: ', min_x)
print('Max x_key value is: ', max_x)
min_y = (kmeans_df[y_key].min())
max_y = (kmeans_df[y_key].max())
print('The y_key value is: ', y_key)
print('Min y_key value is: ', min_y)
print('Max y_key value is: ', max_y)
min_z = (kmeans_df[z_key].min())
max_z = (kmeans_df[z_key].max())
print('The y_key value is: ', z_key)
print('Min y_key value is: ', min_z)
print('Max y_key value is: ', max_z)

#ASSIGN NUMBER OF k POINTS (CLUSTERS) & RANDOMLY DEFINE CO-ORDINATES BASED ON MAX/MIN KEY VALUES
print('***************************************K DATA*************************************')
K_positions = np.zeros((k_points, 3))
for i in range(0,k_points):
        K_positions[i,0] = np.random.uniform(low=min_x, high=max_x)
        K_positions[i,1] = np.random.uniform(low=min_y, high=max_y)
        K_positions[i,2] = np.random.uniform(low=min_z, high=max_z)
        print('k',i,' guess co-ordinate is: ','(',K_positions[i,0],',',K_positions[i,1],',',K_positions[i,2],')')
x = kmeans_df.loc[:, x_key]
y = kmeans_df.loc[:, y_key]
z = kmeans_df.loc[:, z_key]
interest =  kmeans_df.loc[:, interest_key]
x_k = K_positions[:,0]
y_k = K_positions[:,1]
z_k = K_positions[:,2]

#PLOT INITIAL k POINT POSITIONS and DATAFRAME
print('*************************************PLOT K-MEANS INITIAL***********************************')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c = 'grey')
ax.scatter(x_k,y_k,z_k, c='k', marker= '*')
plt.title('INITIAL K-MEAN ASSIGNMENT')
ax.set_xlabel(x_key)
ax.set_ylabel(y_key)
ax.set_zlabel(z_key)
plt.show()

#PLOT INITIAL k POINT POSITIONS and DATAFRAME
print('*************************************UPDATE K-MEAN***********************************')
dist_tracker = np.zeros(k_points)
k_tracker = np.zeros(len(kmeans_df))
centroidsx = np.zeros(k_points)
centroidsy = np.zeros(k_points)
centroidsz = np.zeros(k_points)
k_assignment = np.zeros(k_points)
for n in range(0,1000):
    for i in range(0,len(kmeans_df)):
        for j in range(0,k_points):
            dist_tracker[j] = np.sqrt((x[i]- K_positions[j,0])**2 + (y[i]- K_positions[j,1])**2 + (z[i]- K_positions[j,2])**2)
            #print(j,' dist_tracker value = ',dist_tracker[j])
        for k in range(0,k_points):
                if dist_tracker[k] == np.amin(dist_tracker):
                    #print(k,' selected minimum',np.amin(dist_tracker))
                    centroidsx[k] = centroidsx[k] + x[i]
                    centroidsy[k] = centroidsy[k] + y[i]
                    centroidsz[k] = centroidsz[k] + z[i]
                    k_tracker[i] = k+1
                    k_assignment[k] = k_assignment[k] + 1
                #print('k_assignment ',k,' =',k_assignment[k])
    for i in range(0,k_points):
        #print('X Centroid ',i,' =',centroidsx[i])
        if k_assignment[i] <= min_assign:
            k_assignment[i] = 1
            centroidsx[i] = np.random.uniform(low=min_x, high=max_x)
            centroidsy[i] = np.random.uniform(low=min_y, high=max_y)
            centroidsz[i] = np.random.uniform(low=min_z, high=max_z)
        K_positions[i,0] = centroidsx[i]/float(k_assignment[i])
        K_positions[i,1] = centroidsy[i]/float(k_assignment[i])
        K_positions[i,2] = centroidsz[i]/float(k_assignment[i])
    for q in range(0,k_points):
        centroidsx[q] = 0
        centroidsy[q] = 0
        centroidsz[q] = 0
        k_assignment[q] = 0
        #k_tracker[q] = 0
    #print(k_assignment)
print('k_tracker:')
print(k_tracker)

print('*************************************PLOT K-MEANS FINAL***********************************')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(x_key)
ax.set_ylabel(y_key)
ax.set_zlabel(z_key)
plt.title('FINAL K-MEAN ASSIGNMENT')
for i in range(0,len(kmeans_df)):
    if k_tracker[i] == 1.0:
        ax.scatter(x[i], y[i], z[i], c='b')
    if k_tracker[i] == 2.0:
        ax.scatter(x[i], y[i], z[i], c='r')
    if k_tracker[i] == 3.0:
        ax.scatter(x[i], y[i], z[i], c='g')
    if k_tracker[i] == 4.0:
        ax.scatter(x[i], y[i], z[i], c='magenta')
    if k_tracker[i] == 5.0:
        ax.scatter(x[i], y[i], z[i], c='cyan')
    if k_tracker[i] == 6.0:
        ax.scatter(x[i], y[i], z[i], c='blueviolet')
    if k_tracker[i] == 7.0:
        ax.scatter(x[i], y[i], z[i], c='orange')
    if k_tracker[i] == 8.0:
        ax.scatter(x[i], y[i], z[i], c='yellow')
    if k_tracker[i] == 9.0:
        ax.scatter(x[i], y[i], z[i], c='palegreen')
    if k_tracker[i] == 10.0:
        ax.scatter(x[i], y[i], z[i], c='grey') 
ax.scatter(x_k, y_k, z_k, c='k', marker= '*')
plt.show()

print('*************************************SUMMARISE CLUSTER & INTEREST DATA***********************************')


cluster_interest_arr = pd.DataFrame( np.zeros((10,10)), columns = range(1,11))
# Each row is a cluster and each column is an interst
'''
loop through length of dataset
for each tracker element, print properties and increment count by 1
'''

unique, counts = np.unique(k_tracker, return_counts=True)

for i,x in enumerate(k_tracker):
    print('Index', i, 'belongs to cluster:', x , ' and has ', interest_key, interest[i])
    cluster_interest_arr.iloc[int(x-1), int(interest[i]-1)] += 1

print('')

for j in range(len(unique)):
    print("Cluster ", unique[j], " contains:", counts[j])
    if j == len(unique)-1: print("No more clusters found")

print('')
print('*************************************ARRAY OF CLUSTER & INTEREST DATA***********************************')
print('')
print(cluster_interest_arr)

