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
#CLUSTER TOTAL DATA
cluster_1 = 0
cluster_2 = 0
cluster_3 = 0
cluster_4 = 0
cluster_5 = 0
cluster_6 = 0
cluster_7 = 0
cluster_8 = 0
cluster_9 = 0
cluster_10 = 0

#CLUSTER 1 INTEREST DATA
cluster_1_interest_1 = 0
cluster_1_interest_2 = 0
cluster_1_interest_3 = 0
cluster_1_interest_4 = 0
cluster_1_interest_5 = 0
cluster_1_interest_6 = 0
cluster_1_interest_7 = 0
cluster_1_interest_8 = 0
cluster_1_interest_9 = 0
cluster_1_interest_10 = 0
#CLUSTER 2 INTEREST DATA
cluster_2_interest_1 = 0
cluster_2_interest_2 = 0
cluster_2_interest_3 = 0
cluster_2_interest_4 = 0
cluster_2_interest_5 = 0
cluster_2_interest_6 = 0
cluster_2_interest_7 = 0
cluster_2_interest_8 = 0
cluster_2_interest_9 = 0
cluster_2_interest_10 = 0
#CLUSTER 3 INTEREST DATA
cluster_3_interest_1 = 0
cluster_3_interest_2 = 0
cluster_3_interest_3 = 0
cluster_3_interest_4 = 0
cluster_3_interest_5 = 0
cluster_3_interest_6 = 0
cluster_3_interest_7 = 0
cluster_3_interest_8 = 0
cluster_3_interest_9 = 0
cluster_3_interest_10 = 0
#CLUSTER 4 INTEREST DATA
cluster_4_interest_1 = 0
cluster_4_interest_2 = 0
cluster_4_interest_3 = 0
cluster_4_interest_4 = 0
cluster_4_interest_5 = 0
cluster_4_interest_6 = 0
cluster_4_interest_7 = 0
cluster_4_interest_8 = 0
cluster_4_interest_9 = 0
cluster_4_interest_10 = 0
#CLUSTER 5 INTEREST DATA
cluster_5_interest_1 = 0
cluster_5_interest_2 = 0
cluster_5_interest_3 = 0
cluster_5_interest_4 = 0
cluster_5_interest_5 = 0
cluster_5_interest_6 = 0
cluster_5_interest_7 = 0
cluster_5_interest_8 = 0
cluster_5_interest_9 = 0
cluster_5_interest_10 = 0
#CLUSTER 6 INTEREST DATA
cluster_6_interest_1 = 0
cluster_6_interest_2 = 0
cluster_6_interest_3 = 0
cluster_6_interest_4 = 0
cluster_6_interest_5 = 0
cluster_6_interest_6 = 0
cluster_6_interest_7 = 0
cluster_6_interest_8 = 0
cluster_6_interest_9 = 0
cluster_6_interest_10 = 0
#CLUSTER 7 INTEREST DATA
cluster_7_interest_1 = 0
cluster_7_interest_2 = 0
cluster_7_interest_3 = 0
cluster_7_interest_4 = 0
cluster_7_interest_5 = 0
cluster_7_interest_6 = 0
cluster_7_interest_7 = 0
cluster_7_interest_8 = 0
cluster_7_interest_9 = 0
cluster_7_interest_10 = 0
#CLUSTER 8 INTEREST DATA
cluster_8_interest_1 = 0
cluster_8_interest_2 = 0
cluster_8_interest_3 = 0
cluster_8_interest_4 = 0
cluster_8_interest_5 = 0
cluster_8_interest_6 = 0
cluster_8_interest_7 = 0
cluster_8_interest_8 = 0
cluster_8_interest_9 = 0
cluster_8_interest_10 = 0
#CLUSTER 9 INTEREST DATA
cluster_9_interest_1 = 0
cluster_9_interest_2 = 0
cluster_9_interest_3 = 0
cluster_9_interest_4 = 0
cluster_9_interest_5 = 0
cluster_9_interest_6 = 0
cluster_9_interest_7 = 0
cluster_9_interest_8 = 0
cluster_9_interest_9 = 0
cluster_9_interest_10 = 0
#CLUSTER 10 INTEREST DATA
cluster_10_interest_1 = 0
cluster_10_interest_2 = 0
cluster_10_interest_3 = 0
cluster_10_interest_4 = 0
cluster_10_interest_5 = 0
cluster_10_interest_6 = 0
cluster_10_interest_7 = 0
cluster_10_interest_8 = 0
cluster_10_interest_9 = 0
cluster_10_interest_10 = 0

#TRACK TOTAL NUMBER OF DATA POINTS IN CLUSTER AND HOW MANY OF EACH INTERST ARE IN A CLUSTER
for i in range(0,len(kmeans_df)):
    if k_tracker[i] == 1:
        print('Index', i, 'belongs to cluster:', 1, ' and has ', interest_key, interest[i])
        cluster_1 = cluster_1 + 1
        if interest[i] == 1:
            cluster_1_interest_1 = cluster_1_interest_1 + 1
        if interest[i] == 2:
            cluster_1_interest_2 = cluster_1_interest_2 + 1
        if interest[i] == 3:
            cluster_1_interest_3 = cluster_1_interest_3 + 1
        if interest[i] == 4:
            cluster_1_interest_4 = cluster_1_interest_4 + 1
        if interest[i] == 5:
            cluster_1_interest_5 = cluster_1_interest_5 + 1
        if interest[i] == 6:
            cluster_1_interest_6 = cluster_1_interest_6 + 1
        if interest[i] == 7:
            cluster_1_interest_7 = cluster_1_interest_7 + 1
        if interest[i] == 8:
            cluster_1_interest_8 = cluster_1_interest_8 + 1
        if interest[i] == 9:
            cluster_1_interest_9 = cluster_1_interest_9 + 1
        if interest[i] == 10:
            cluster_1_interest_10 = cluster_1_interest_10 + 1
    if k_tracker[i] == 2:
        print('Index', i, 'belongs to cluster:', 2, ' and has ', interest_key, interest[i])
        cluster_2 = cluster_2 + 1
        if interest[i] == 1:
            cluster_2_interest_1 = cluster_2_interest_1 + 1
        if interest[i] == 2:
            cluster_2_interest_2 = cluster_2_interest_2 + 1
        if interest[i] == 3:
            cluster_2_interest_3 = cluster_2_interest_3 + 1
        if interest[i] == 4:
            cluster_2_interest_4 = cluster_2_interest_4 + 1
        if interest[i] == 5:
            cluster_2_interest_5 = cluster_2_interest_5 + 1
        if interest[i] == 6:
            cluster_2_interest_6 = cluster_2_interest_6 + 1
        if interest[i] == 7:
            cluster_2_interest_7 = cluster_2_interest_7 + 1
        if interest[i] == 8:
            cluster_2_interest_8 = cluster_2_interest_8 + 1
        if interest[i] == 9:
            cluster_2_interest_9 = cluster_2_interest_9 + 1
        if interest[i] == 10:
            cluster_2_interest_10 = cluster_2_interest_10 + 1
    if k_tracker[i] == 3:
        print('Index', i, 'belongs to cluster:', 3, ' and has ', interest_key, interest[i])
        cluster_3 = cluster_3 + 1
        if interest[i] == 1:
            cluster_3_interest_1 = cluster_3_interest_1 + 1
        if interest[i] == 2:
            cluster_3_interest_2 = cluster_3_interest_2 + 1
        if interest[i] == 3:
            cluster_3_interest_3 = cluster_3_interest_3 + 1
        if interest[i] == 4:
            cluster_3_interest_4 = cluster_3_interest_4 + 1
        if interest[i] == 5:
            cluster_3_interest_5 = cluster_3_interest_5 + 1
        if interest[i] == 6:
            cluster_3_interest_6 = cluster_3_interest_6 + 1
        if interest[i] == 7:
            cluster_3_interest_7 = cluster_3_interest_7 + 1
        if interest[i] == 8:
            cluster_3_interest_8 = cluster_3_interest_8 + 1
        if interest[i] == 9:
            cluster_3_interest_9 = cluster_3_interest_9 + 1
        if interest[i] == 10:
            cluster_3_interest_10 = cluster_3_interest_10 + 1
    if k_tracker[i] == 4:
        print('Index', i, 'belongs to cluster:', 4, ' and has ', interest_key, interest[i])
        cluster_4 = cluster_4 + 1
        if interest[i] == 1:
            cluster_4_interest_1 = cluster_4_interest_1 + 1
        if interest[i] == 2:
            cluster_4_interest_2 = cluster_4_interest_2 + 1
        if interest[i] == 3:
            cluster_4_interest_3 = cluster_4_interest_3 + 1
        if interest[i] == 4:
            cluster_4_interest_4 = cluster_4_interest_4 + 1
        if interest[i] == 5:
            cluster_4_interest_5 = cluster_4_interest_5 + 1
        if interest[i] == 6:
            cluster_4_interest_6 = cluster_4_interest_6 + 1
        if interest[i] == 7:
            cluster_4_interest_7 = cluster_4_interest_7 + 1
        if interest[i] == 8:
            cluster_4_interest_8 = cluster_4_interest_8 + 1
        if interest[i] == 9:
            cluster_4_interest_9 = cluster_4_interest_9 + 1
        if interest[i] == 10:
            cluster_4_interest_10 = cluster_4_interest_10 + 1
    if k_tracker[i] == 5:
        print('Index', i, 'belongs to cluster:', 5, ' and has ', interest_key, interest[i])
        cluster_5 = cluster_5 + 1
        if interest[i] == 1:
            cluster_5_interest_1 = cluster_5_interest_1 + 1
        if interest[i] == 2:
            cluster_5_interest_2 = cluster_5_interest_2 + 1
        if interest[i] == 3:
            cluster_5_interest_3 = cluster_5_interest_3 + 1
        if interest[i] == 4:
            cluster_5_interest_4 = cluster_5_interest_4 + 1
        if interest[i] == 5:
            cluster_5_interest_5 = cluster_5_interest_5 + 1
        if interest[i] == 6:
            cluster_5_interest_6 = cluster_5_interest_6 + 1
        if interest[i] == 7:
            cluster_5_interest_7 = cluster_5_interest_7 + 1
        if interest[i] == 8:
            cluster_5_interest_8 = cluster_5_interest_8 + 1
        if interest[i] == 9:
            cluster_5_interest_9 = cluster_5_interest_9 + 1
        if interest[i] == 10:
            cluster_5_interest_10 = cluster_5_interest_10 + 1
    if k_tracker[i] == 6:
        print('Index', i, 'belongs to cluster:', 6, ' and has ', interest_key, interest[i])
        cluster_6 = cluster_6 + 1
        if interest[i] == 1:
            cluster_6_interest_1 = cluster_6_interest_1 + 1
        if interest[i] == 2:
            cluster_6_interest_2 = cluster_6_interest_2 + 1
        if interest[i] == 3:
            cluster_6_interest_3 = cluster_6_interest_3 + 1
        if interest[i] == 4:
            cluster_6_interest_4 = cluster_6_interest_4 + 1
        if interest[i] == 5:
            cluster_6_interest_5 = cluster_6_interest_5 + 1
        if interest[i] == 6:
            cluster_6_interest_6 = cluster_6_interest_6 + 1
        if interest[i] == 7:
            cluster_6_interest_7 = cluster_6_interest_7 + 1
        if interest[i] == 8:
            cluster_6_interest_8 = cluster_6_interest_8 + 1
        if interest[i] == 9:
            cluster_6_interest_9 = cluster_6_interest_9 + 1
        if interest[i] == 10:
            cluster_6_interest_10 = cluster_6_interest_10 + 1
    if k_tracker[i] == 7:
        print('Index', i, 'belongs to cluster:', 7, ' and has ', interest_key, interest[i])
        cluster_7 = cluster_7 + 1
        if interest[i] == 1:
            cluster_7_interest_1 = cluster_7_interest_1 + 1
        if interest[i] == 2:
            cluster_7_interest_2 = cluster_7_interest_2 + 1
        if interest[i] == 3:
            cluster_7_interest_3 = cluster_7_interest_3 + 1
        if interest[i] == 4:
            cluster_7_interest_4 = cluster_7_interest_4 + 1
        if interest[i] == 5:
            cluster_7_interest_5 = cluster_7_interest_5 + 1
        if interest[i] == 6:
            cluster_7_interest_6 = cluster_7_interest_6 + 1
        if interest[i] == 7:
            cluster_7_interest_7 = cluster_7_interest_7 + 1
        if interest[i] == 8:
            cluster_7_interest_8 = cluster_7_interest_8 + 1
        if interest[i] == 9:
            cluster_7_interest_9 = cluster_7_interest_9 + 1
        if interest[i] == 10:
            cluster_7_interest_10 = cluster_7_interest_10 + 1
    if k_tracker[i] == 8:
        print('Index', i, 'belongs to cluster:', 8, ' and has ', interest_key, interest[i])
        cluster_8 = cluster_8 + 1
        if interest[i] == 1:
            cluster_8_interest_1 = cluster_8_interest_1 + 1
        if interest[i] == 2:
            cluster_8_interest_2 = cluster_8_interest_2 + 1
        if interest[i] == 3:
            cluster_8_interest_3 = cluster_8_interest_3 + 1
        if interest[i] == 4:
            cluster_8_interest_4 = cluster_8_interest_4 + 1
        if interest[i] == 5:
            cluster_8_interest_5 = cluster_8_interest_5 + 1
        if interest[i] == 6:
            cluster_8_interest_6 = cluster_8_interest_6 + 1
        if interest[i] == 7:
            cluster_8_interest_7 = cluster_8_interest_7 + 1
        if interest[i] == 8:
            cluster_8_interest_8 = cluster_8_interest_8 + 1
        if interest[i] == 9:
            cluster_8_interest_9 = cluster_8_interest_9 + 1
        if interest[i] == 10:
            cluster_8_interest_10 = cluster_8_interest_10 + 1
    if k_tracker[i] == 9:
        print('Index', i, 'belongs to cluster:', 9, ' and has ', interest_key, interest[i])
        cluster_9 = cluster_9 + 1
        if interest[i] == 1:
            cluster_9_interest_1 = cluster_9_interest_1 + 1
        if interest[i] == 2:
            cluster_9_interest_2 = cluster_9_interest_2 + 1
        if interest[i] == 3:
            cluster_9_interest_3 = cluster_9_interest_3 + 1
        if interest[i] == 4:
            cluster_9_interest_4 = cluster_9_interest_4 + 1
        if interest[i] == 5:
            cluster_9_interest_5 = cluster_9_interest_5 + 1
        if interest[i] == 6:
            cluster_9_interest_6 = cluster_9_interest_6 + 1
        if interest[i] == 7:
            cluster_9_interest_7 = cluster_9_interest_7 + 1
        if interest[i] == 8:
            cluster_9_interest_8 = cluster_9_interest_8 + 1
        if interest[i] == 9:
            cluster_9_interest_9 = cluster_9_interest_9 + 1
        if interest[i] == 10:
            cluster_9_interest_10 = cluster_9_interest_10 + 1
            cluster_8_interest_10 = cluster_8_interest_10 + 1
    if k_tracker[i] == 10:
        print('Index', i, 'belongs to cluster:', 9, ' and has ', interest_key, interest[i])
        cluster_10 = cluster_10 + 1
        if interest[i] == 1:
            cluster_10_interest_1 = cluster_10_interest_1 + 1
        if interest[i] == 2:
            cluster_10_interest_2 = cluster_10_interest_2 + 1
        if interest[i] == 3:
            cluster_10_interest_3 = cluster_10_interest_3 + 1
        if interest[i] == 4:
            cluster_10_interest_4 = cluster_10_interest_4 + 1
        if interest[i] == 5:
            cluster_10_interest_5 = cluster_10_interest_5 + 1
        if interest[i] == 6:
            cluster_10_interest_6 = cluster_10_interest_6 + 1
        if interest[i] == 7:
            cluster_10_interest_7 = cluster_10_interest_7 + 1
        if interest[i] == 8:
            cluster_10_interest_8 = cluster_10_interest_8 + 1
        if interest[i] == 9:
            cluster_10_interest_9 = cluster_10_interest_9 + 1
        if interest[i] == 10:
            cluster_10_interest_10 = cluster_10_interest_10 + 1

#PRINT NUMBER OF POINTS IN EACH CLUSTER
print(' ')       
print('cluster 1 contains: ' ,cluster_1)
print('cluster 2 contains: ' ,cluster_2)
print('cluster 3 contains: ' ,cluster_3)
print('cluster 4 contains: ' ,cluster_4)
print('cluster 5 contains: ' ,cluster_5)
print('cluster 6 contains: ' ,cluster_6)
print('cluster 7 contains: ' ,cluster_7)
print('cluster 8 contains: ' ,cluster_8)
print('cluster 9 contains: ' ,cluster_9)
print('cluster 10 contains: ',cluster_10)

#PRINT NUMBER OF EACH INTEREST IN CLUSTER
print(' ')
print('Cluster 1 contains:')
print(cluster_1_interest_1, interest_key, 1)
print(cluster_1_interest_2, interest_key, 2)
print(cluster_1_interest_3, interest_key, 3)
print(cluster_1_interest_4, interest_key, 4)
print(cluster_1_interest_5, interest_key, 5)
print(cluster_1_interest_6, interest_key, 6)
print(cluster_1_interest_7, interest_key, 7)
print(cluster_1_interest_8, interest_key, 8)
print(cluster_1_interest_9, interest_key, 9)
print(cluster_1_interest_10, interest_key, 10)
print(' ')
print('Cluster 2 contains:')
print(cluster_2_interest_1, interest_key, 1)
print(cluster_2_interest_2, interest_key, 2)
print(cluster_2_interest_3, interest_key, 3)
print(cluster_2_interest_4, interest_key, 4)
print(cluster_2_interest_5, interest_key, 5)
print(cluster_2_interest_6, interest_key, 6)
print(cluster_2_interest_7, interest_key, 7)
print(cluster_2_interest_8, interest_key, 8)
print(cluster_2_interest_9, interest_key, 9)
print(cluster_2_interest_10, interest_key, 10)
print(' ')
print('Cluster 3 contains:')
print(cluster_3_interest_1, interest_key, 1)
print(cluster_3_interest_2, interest_key, 2)
print(cluster_3_interest_3, interest_key, 3)
print(cluster_3_interest_4, interest_key, 4)
print(cluster_3_interest_5, interest_key, 5)
print(cluster_3_interest_6, interest_key, 6)
print(cluster_3_interest_7, interest_key, 7)
print(cluster_3_interest_8, interest_key, 8)
print(cluster_3_interest_9, interest_key, 9)
print(cluster_3_interest_10, interest_key, 10)
print(' ')
print('Cluster 4 contains:')
print(cluster_4_interest_1, interest_key, 1)
print(cluster_4_interest_2, interest_key, 2)
print(cluster_4_interest_3, interest_key, 3)
print(cluster_4_interest_4, interest_key, 4)
print(cluster_4_interest_5, interest_key, 5)
print(cluster_4_interest_6, interest_key, 6)
print(cluster_4_interest_7, interest_key, 7)
print(cluster_4_interest_8, interest_key, 8)
print(cluster_4_interest_9, interest_key, 9)
print(cluster_4_interest_10, interest_key, 10)
print(' ')
print('Cluster 5 contains:')
print(cluster_5_interest_1, interest_key, 1)
print(cluster_5_interest_2, interest_key, 2)
print(cluster_5_interest_3, interest_key, 3)
print(cluster_5_interest_4, interest_key, 4)
print(cluster_5_interest_5, interest_key, 5)
print(cluster_5_interest_6, interest_key, 6)
print(cluster_5_interest_7, interest_key, 7)
print(cluster_5_interest_8, interest_key, 8)
print(cluster_5_interest_9, interest_key, 9)
print(cluster_5_interest_10, interest_key, 10)
print(' ')
print('Cluster 6 contains:')
print(cluster_6_interest_1, interest_key, 1)
print(cluster_6_interest_2, interest_key, 2)
print(cluster_6_interest_3, interest_key, 3)
print(cluster_6_interest_4, interest_key, 4)
print(cluster_6_interest_5, interest_key, 5)
print(cluster_6_interest_6, interest_key, 6)
print(cluster_6_interest_7, interest_key, 7)
print(cluster_6_interest_8, interest_key, 8)
print(cluster_6_interest_9, interest_key, 9)
print(cluster_6_interest_10, interest_key, 10)
print(' ')
print('Cluster 7 contains:')
print(cluster_7_interest_1, interest_key, 1)
print(cluster_7_interest_2, interest_key, 2)
print(cluster_7_interest_3, interest_key, 3)
print(cluster_7_interest_4, interest_key, 4)
print(cluster_7_interest_5, interest_key, 5)
print(cluster_7_interest_6, interest_key, 6)
print(cluster_7_interest_7, interest_key, 7)
print(cluster_7_interest_8, interest_key, 8)
print(cluster_7_interest_9, interest_key, 9)
print(cluster_7_interest_10, interest_key, 10)
print(' ')
print('Cluster 8 contains:')
print(cluster_8_interest_1, interest_key, 1)
print(cluster_8_interest_2, interest_key, 2)
print(cluster_8_interest_3, interest_key, 3)
print(cluster_8_interest_4, interest_key, 4)
print(cluster_8_interest_5, interest_key, 5)
print(cluster_8_interest_6, interest_key, 6)
print(cluster_8_interest_7, interest_key, 7)
print(cluster_8_interest_8, interest_key, 8)
print(cluster_8_interest_9, interest_key, 9)
print(cluster_8_interest_10, interest_key, 10)
print(' ')
print('Cluster 9 contains:')
print(cluster_9_interest_1, interest_key, 1)
print(cluster_9_interest_2, interest_key, 2)
print(cluster_9_interest_3, interest_key, 3)
print(cluster_9_interest_4, interest_key, 4)
print(cluster_9_interest_5, interest_key, 5)
print(cluster_9_interest_6, interest_key, 6)
print(cluster_9_interest_7, interest_key, 7)
print(cluster_9_interest_8, interest_key, 8)
print(cluster_9_interest_9, interest_key, 9)
print(cluster_9_interest_10, interest_key, 10)
print(' ')
print('Cluster 10 contains:')
print(cluster_10_interest_1, interest_key, 1)
print(cluster_10_interest_2, interest_key, 2)
print(cluster_10_interest_3, interest_key, 3)
print(cluster_10_interest_4, interest_key, 4)
print(cluster_10_interest_5, interest_key, 5)
print(cluster_10_interest_6, interest_key, 6)
print(cluster_10_interest_7, interest_key, 7)
print(cluster_10_interest_8, interest_key, 8)
print(cluster_10_interest_9, interest_key, 9)
print(cluster_10_interest_10, interest_key, 10)

