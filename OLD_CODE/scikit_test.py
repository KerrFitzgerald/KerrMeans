from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def df_read(datafile):
    kmeans_df = pd.read_csv(datafile, delimiter = ',')                  #PANDAS INBUILT FUNCTION TO READ DATAFRAME
    return kmeans_df


wine_df = df_read('wine.data')

x = wine_df.loc[:, 'Alcohol']                    #X DATAPOINT POSITIONS
y = wine_df.loc[:, 'Total_phenols']                    #Y DATAPOINT POSITIONS
z = wine_df.loc[:, 'Hue']                    #Y DATAPOINT POSITIONS

use_array = np.empty([178, 3])

for i in range(0,len(x)):
    use_array[i,0] = x[i]
    use_array[i,1] = y[i]
    use_array[i,2] = z[i]
    
#print(x, y , z)
#print(use_array)


kmeans = KMeans(n_clusters=3, random_state=0).fit(use_array)
k_tracker = kmeans.labels_


unique, counts = np.unique(k_tracker, return_counts=True)
count_dict = dict(zip(unique, counts))
print(count_dict)
#kmeans.predict([[0, 0], [12, 3]])
print('Cluster Centres\n',kmeans.cluster_centers_)
print(k_tracker)


x_k = kmeans.cluster_centers_[:,0]                                              #X MEAN POSITIONS
y_k = kmeans.cluster_centers_[:,1]                                              #Y MEAN POSITIONS
z_k = kmeans.cluster_centers_[:,2]

color_list = ['b', 'r', 'g', 'magenta', 'cyan', 'blueviolet',\
'orange', 'yellow', 'palegreen', 'grey', 'lime', 'peru', \
'teal', 'hotpink', 'cornflowerblue', 'lightcoral', 'darkgray',\
'whitesmoke', 'rosybrown', 'firebrick', 'salmon', 'chocolate',\
'bisque', 'tan', 'gold', 'olive', 'honeydew','thistle', 'k'] 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')                          #CREATE 3D PLOT
ax.scatter(x_k, y_k, z_k, marker='*', color='k')
for i in range(0,len(x)):                                  #CYCLE THROUGH DATAPOINTS
    for k in range(0,(3)):                           #CYCLE THROUGH CLUSTERS
        if k_tracker[i] == k:                                       #IF DATAPOINT BELONGS TO CLUSTER K ('k_tracker' IS USED TO DETERMINE THIS)
            ax.scatter(x[i], y[i], z[i],c= color_list[k])
plt.show()