#PYTHON PROGRAM FOR CLUSTERING 3D DATA. CREATED BY KERR FITZGERALD 2020.
#PROGRAM IS BASED ON ALGORITHM 20.7 FROM "INFORMATION THEORY, INFERENCE, AND LEARNING ALGORITHMS" BY DAVID J.C. MACKAY
#THE PROGRAM IMPLEMENTS THE SOFT k-MEANS ALGORITHM. 

#IMPORT RELEVANT PYTHON MODULES
import pandas as pd
import sys
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#USER INPUT SELECTED FROM COMMAND LINE
#(Create function from this using dictionary for number of dimensions?)
cla = sys.argv
datafile = cla[1]
k_points = int(cla[2])
x_key = cla[3]
y_key = cla[4]
z_key = cla[5]
interest_key =cla[6]
iterations = int(cla[7])
#min_assign = int(cla[8])
beta = float(cla[8])
plot = cla[9]

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

#SUMMARISE INPUT/SELCTED KEY DATA FOR USER
print('**************************************INPUT/SELCTED KEY************************************')
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
print('Max interations is: ', iterations)
print('Beta value is     : ', beta)
if plot == 'yes':
    print('Plotting turned on')
else:
    print('Plotting turned off')


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
if plot == 'yes':
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
resp_tracker = np.zeros((len(kmeans_df),k_points))
resp_tracker2 = np.zeros((len(kmeans_df),k_points))
#print(resp_tracker)
#print(resp_tracker)
dist_tracker = np.zeros(k_points)
k_tracker = np.zeros(len(kmeans_df))
k_assignment = np.zeros(k_points)
for n in range(0,iterations):
    for i in range(0,len(kmeans_df)):
        for j in range(0,k_points):
            dist_tracker[j] = np.sqrt((x[i]- K_positions[j,0])**2 + (y[i]- K_positions[j,1])**2 + (z[i]- K_positions[j,2])**2)
            resp_tracker[i,j] = np.exp(-beta*((dist_tracker[j])))
    temp = resp_tracker.sum(axis=1)
    for i in range(0,len(kmeans_df)):
        for j in range(0,k_points):
            resp_tracker2[i,j] = (resp_tracker[i,j]/temp[i])
    temp = resp_tracker2.sum(axis=0)
    for j in range(0,k_points):
        #NEED TO SET LOOP SO THAT IF EACH K-POINTS DOESNT MOVE MORE THAN SOME SMALL DISTANCE E>G )>0000000.1 then exit loop
        total_x = 0
        total_y = 0
        total_z = 0
        for i in range(0,len(kmeans_df)):
            rkx = resp_tracker2[i,j] * x[i]
            rky = resp_tracker2[i,j] * y[i]
            rkz = resp_tracker2[i,j] * z[i]
            total_x = total_x + rkx
            total_y = total_y + rky
            total_z = total_z + rkz           
        K_positions[j,0] = total_x/temp[j]
        K_positions[j,1] = total_y/temp[j]
        K_positions[j,2] = total_z/temp[j]       
x_k = K_positions[:,0]
y_k = K_positions[:,1]
z_k = K_positions[:,2]
print("Dist_tracker", dist_tracker)

print('*************************************PLOT K-MEANS FINAL***********************************')
color_list = ['b', 'r', 'g', 'magenta', 'cyan', 'blueviolet', 'orange', 'yellow', 'palegreen', 'grey']
ClusterDict={}
for i in range(1,k_points+1):
        ClusterDict["Cluster {0}".format(i)]=0
ClusterList = sorted(ClusterDict.keys())
for i in range(0,len(kmeans_df)):
    current_max = 0.0
    for j in range(0, k_points):
        if resp_tracker2[i,j] > current_max:
            current_max = resp_tracker2[i,j]
            k_tracker[i] = j + 1

if plot == 'yes':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_zlabel(z_key)
    plt.title('FINAL K-MEAN ASSIGNMENT')
    for i in range(0,len(kmeans_df)):
        for j in range(1,k_points+1):
            if k_tracker[i] == j:
                ax.scatter(x[i], y[i], z[i], c= color_list[j-1])# label = ClusterList[j-1])
    ax.scatter(x_k, y_k, z_k, c='k', marker= '*')
    plt.legend(loc="upper left")
    plt.show()

print('*************************************SUMMARISE CLUSTER & INTEREST DATA***********************************')

unique, counts = np.unique(interest, return_counts=True)
interest_total = len(dict(zip(unique, counts)))
cluster_interest_arr = pd.DataFrame( np.zeros((k_points,interest_total)), columns = range(1,interest_total+1))
# EACH ROW IS A CLUSTER AND EACH COLUMN IS AN INTEREST PARAMETER
# LOOP THROUGH LENGTH OF DATASET
# FOR EACH TRACKER ELEMENT PRINT PROPERTIES AND INCREMENT COUNT BY 1
unique, counts = np.unique(k_tracker, return_counts=True)
for i,x in enumerate(k_tracker):
    j = int(k_tracker[i]-1)
    print('Index', i, 'belongs to cluster:', x , ' and has ', interest_key, interest[i], "with score ",resp_tracker2[i,j])
    cluster_interest_arr.iloc[int(x-1), int(interest[i]-1)] += 1
print('')
for j in range(len(unique)):
    print("Cluster ", unique[j], " contains:", counts[j])
    if j == len(unique)-1: print("No more clusters found")
print('')
print('*************************************ARRAY OF CLUSTER & INTEREST DATA***********************************')
print('NOTE: Rows are clusters and the columns are the interest parameters')
print('')
print(cluster_interest_arr)

