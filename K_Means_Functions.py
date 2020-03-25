#PYTHON FUNCTIONS FOR SOFT K-MEANS CLUSTERING OF N-DIMENSIONAL DATA.
#FUNCTIONS ARE BASED ON ALGORITHMS IN "INFORMATION THEORY, INFERENCE, & LEARNING ALGORITHMS" BY DAVID J.C. MACKAY
#CREATED BY KERR FITZGERALD, THOMAS BENNETT & AIDEN PEAKMAN 2020.

#IMPORT RELEVANT PYTHON MODULES
import pandas as pd
import sys
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#***************************COMMON FUNCTIONS****************************
#READ USER INPUT FROM COMMAND LINE
def input_read():
    cla = sys.argv
    datafile = str(cla[1])                                              #USER SPECIFIED FILENAME
    n_clustr = int(cla[2])                                              #USER SPECIFIED NUMBER OF CLUSTERS
    n_dimens = int(len(cla) - 8)                                        #ALLOWS AUTOMATIC DETECTION OF NUMBER OF DIMENSIONS FROM USER INPUT
    interest = str(cla[n_dimens + 3])                                   #USER SPECIFIED INTEREST PARAMETER
    max_iter = int(cla[n_dimens + 4])                                   #USER SPECIFIED MAXIMUM ITERATIONS
    set_norm = str(cla[n_dimens + 5])                                   #NORMALISE DATAFRAME COLUMNS
    betainit = float(cla[n_dimens + 6])                                 #USER SPECIFIED STARTING VALUE OF BETA
    min_pnts = int(cla[n_dimens + 7])                                   #USER SPECIFIED MINIMUM NUMBER OF POINTS ALLOWED PER CLUSTER (SET TO -1 TO TURN OFF)
    DimsDict = {}                                                       #CREATE DICTIONARY WHICH CONTAINS DIMi AS THE 'KEYS' & USER SPECIFIED INPUT AS 'VALUES'
    for i in range(1,n_dimens+1):
        DimsDict["Dim{0}".format(i)] = cla[i+2]
    return datafile, n_clustr, n_dimens, interest,DimsDict, max_iter,\
           set_norm, betainit, min_pnts
        
#READ IN DATAFRAME FROM CSV FILE (NOTE THAT FIRST ROW OF THE CSV FILE MUST CONTAIN COLUMN HEADERS)
def df_read(datafile):
    kmeans_df = pd.read_csv(datafile, delimiter = ',')                  #PANDAS INBUILT FUNCTION TO READ DATAFRAME
    return kmeans_df

#NORMALISE THE DATAFRAME COLUMNS CONTAINING INTEREST PARAMETERS
def df_normalise(data_frame, param_dict):
    for i in range(0,len(param_dict)):
        param_list = sorted(param_dict.keys())                          #CREATE LIST OF DICTIONARY KEYS (DIMi) THAT CAN BE USED TO ACCESS DATAFRAME COLUMNS
    for i in range(0,int(len(param_list))):                             #CYCLE THROUGH DATAFRAME COLUMNS
        max_interest = (data_frame[param_dict[param_list[i]]].max())    #DETERMINE MAXIMUM VALUE IN THE COLUMN
        data_frame.loc[:,param_dict[param_list[i]]] = \
        data_frame.loc[:,param_dict[param_list[i]]]   \
        * (1.0/max_interest)                                            #NORMALISE COLUMN VALUES BY DIVIDING BY MAXIMUM VALUE
    normalised_data_frame = data_frame
    return normalised_data_frame
    
#ASSIGN INITIAL RANDOM K POINTS
def k_assign(dimensions, k_points, data_frame, param_dict):
    k_points = int(k_points)                                            #NUMBER OF CLUSTERS
    k_pos = np.zeros((int(k_points), int(dimensions)))                  #CREATE ARRAY WITH 'k_point' COLUMNS % 'dimensions' ROWS
    for i in range(0,len(param_dict)):
        param_list = sorted(param_dict.keys())
    for i in range(0,int(dimensions)):                                  #CYCLE THROUGH DATAFRAME COLUMNS
            min_inter = (data_frame[param_dict[param_list[i]]].min())   #DETERMINE MINIMUM VALUE IN A COLUMN
            max_inter = (data_frame[param_dict[param_list[i]]].max())   #DETERMINE MAXIMUM VALUE IN A COLUMN
            for k in range(0,int(k_points)):                            #CYCLE THROUGH EACH DIMENSION 
                k_pos[k,i] = np.random.uniform(low=min_inter, \
                high=max_inter)                                         #ASSIGN EACH DIMENSIONAL POSITION TO A RANDOM NUMBER BETWEEN MAX & MIN VALUES
    return k_pos

#CALCULATE EUCLIDEAN DISTANCE AND STORE DISTANCES AS ARRAY
def eucl_dist_arr(dimensions, k_pos, data_frame, param_dict):     
    param_list = sorted(param_dict.keys())
    total_points = int(len(data_frame))
    k_points = int(len(k_pos))
    distances = np.zeros((total_points, k_points))                      #CREATE ARRAY WITH 'k_points' COLUMNS & 'dimensions' ROWS
    total_dist = 0                                                      #CREATE TOTAL TRACKER
    for i in range(0, total_points):                                    #CYCLE THROUGH EACH POINT IN DATAFRAME
        for k in range(0,k_points):                                     #CYCLE THROUGH EACH CLUSTER
            for j in range(0,dimensions):                               #CYCLE THROUGH EACH DIMENSION
                total_dist = total_dist + \
                (data_frame[param_dict[param_list[j]]][i]\
                - k_pos[k,j])**2                                        #KEEP TRACK OF SQAURED DISTANCE BETWEEN CLUSTER K & POINT FOR EACH DIMENSION
            total_dist = np.sqrt(total_dist)                            #SQUAEROOT TO FIND EUCLIDEAN DISTANCE
            distances[i,k] = total_dist
            total_dist = 0.0                                            #RESET TOTAL TRACKER TO ZERO FOR NEXT DATAPOINT
    return distances

#TRACK WHICH DATAPOINT A CLUSTER BELONGS TO (HIGHEST RESPONSIBILITY)
def k_tracker(k_respons_arr, k_pos):
    k_respons_tracker = np.zeros(len(k_respons_arr))                    #CREATE ARRAY TO TRACK WHICH DATAPOINT CLUSTER ASSIGNMENT
    for i in range(0,len(k_respons_arr)):                               #CYCLE THROUGH EACH DATAPOINT
        current_max = 0.0                                               #SET MAXIMUM RESPONSIBILTY TRACKER
        for k in range(0, len(k_pos)):                                  #CYCLE THROUGH EACH CLUSTERS RESPONSIBILITY SCORE
            if k_respons_arr[i,k] > current_max:                        #IF THE RESPONSIBILITY SCORE HIGHER FOR THE CURRENT CLUSTER
                current_max = k_respons_arr[i,k]                        #SET THE MAXIMUM RESPONSIBILTY TRACKER TO THE CURRENT HIGHEST VALUE
                k_respons_tracker[i] = k + 1                            #SET THE RESPONSIBILITY TRACKER FOR THE DATAPOINT TO CLUSTER NUMBER
    return k_respons_tracker

#**********************VERSION 1 SPECIFIC FUNCTIONS*********************
#USE THE DISTANCE ARRAY TO CREATE RESPONSIBILITY ARRAY      
def k_respons_arr(beta, dist_arr, k_pos):     
    beta = float(beta)
    k_respons_arr = np.exp(-beta*((dist_arr)))                          #USES THE DISTANCE ARRAY AND IMPLEMENTS EXPONENTIAL PART OF ALGORITHM 20.7 
    temp = k_respons_arr.sum(axis=1)                                    #CREATES AN ARRAY CONTAINING SUM OF ALL ROWS (SUM OF RESPONSIBILITY SCORE FOR EACH DATAPOINT)
    for i in range(0,len(dist_arr)):                                    #CYCLE THROUGH EACH DATAPOINT
        for k in range(0, len(k_pos)):                                  #CYCLE THROUGH EACH CLUSTER
            k_respons_arr[i,k] = k_respons_arr[i,k]/temp[i]             #DIVIDE EVERY RESPONSIBILITY SCORE FOR CLUSTER BY SUM OF CURRENT ROW
    return k_respons_arr

#***************************GRAPHICS FUNCTIONS**************************
#PLOT INITIAL K-MEAN LOCATIONS AND 2D DATA
def initial_plot_2D(data_frame, k_locations, dim_dict):
    param_list = sorted(dim_dict.keys())
    fig = plt.figure()
    x = data_frame.loc[:, [dim_dict[param_list[0]]]]                    #X DATAPOINT POSITIONS
    y = data_frame.loc[:, [dim_dict[param_list[1]]]]                    #Y DATAPOINT POSITIONS
    x_k = k_locations[:,0]                                              #X MEAN POSITIONS
    y_k = k_locations[:,1]                                              #Y MEAN POSITIONS
    plt.scatter(x,y, c = 'grey')                                        #PLOT DATAPOINTS IN 2D USING GREY DATAPOINTS
    plt.scatter(x_k, y_k, marker= '*', c = 'k')                         #PLOT MEANS IN 2D USING BLACK STARS
    plt.xlabel([dim_dict[param_list[0]]])                               #SET X-AXIS LABEL
    plt.ylabel([dim_dict[param_list[1]]])                               #SET Y-AXIS LABEL
    plt.title('INITIAL K-MEAN ASSIGNMENT')                              #SET TITLE
    plt.show()

#PLOT FINAL K-MEAN LOCATIONS AND 2D DATA
def final_plot_2D(data_frame, k_locations, dim_dict, k_tracker):
    color_list = ['b', 'r', 'g', 'magenta', 'cyan', 'blueviolet',\
    'orange', 'yellow', 'palegreen', 'grey', 'lime', 'peru', \
    'teal', 'hotpink', 'cornflowerblue', 'lightcoral', 'darkgray',\
    'whitesmoke', 'rosybrown', 'firebrick', 'salmon', 'chocolate',\
    'bisque', 'tan', 'gold', 'olive', 'honeydew','thistle', 'k']        #DEFINE COLOUR LIST FOR 29 CLUSTERS
    param_list = sorted(dim_dict.keys())
    fig = plt.figure()
    x = data_frame.loc[:, [dim_dict[param_list[0]]]]                    #X DATAPOINT POSITIONS
    y = data_frame.loc[:, [dim_dict[param_list[1]]]]                    #Y DATAPOINT POSITIONS
    x_k = k_locations[:,0]                                              #X MEAN POSITIONS
    y_k = k_locations[:,1]                                              #Y MEAN POSITIONS
    for k in range(1,len(k_locations)+1):                               #CYCLE THROUGH CLUSTERS
        print("Cluster", k , color_list[k-1])                           #PRINT COLOUR KEY FOR USER (NEEDED DUE TO DIFFICULTLY IN ADDING LEGEND/CLUSTERS NUMBERS CHANGE EACH RUN)
    for i in range(0,len(data_frame)):                                  #CYCLE THROUGH DATAPOINTS          
        for k in range(1,len(k_locations)+1):                           #CYCLE THROUGH CLUSTERS
            if k_tracker[i] == k:                                       #IF DATAPOINT BELONGS TO CLUSTER K ('k_tracker' IS USED TO DETERMINE THIS)
                plt.scatter(x.iat[i,0], y.iat[i,0], c= color_list[k-1]) #PLOT THE DATAPOINT WHICH APPROPRIATE COLOUR
    plt.scatter(x_k, y_k, marker= '*', c = 'k')                         #PLOT MEANS IN 2D USING BLACK STARS
    plt.xlabel([dim_dict[param_list[0]]])                               #SET X-AXIS LABEL
    plt.ylabel([dim_dict[param_list[1]]])                               #SET Y-AXIS LABEL
    plt.title('FINAL K-MEAN ASSIGNMENT')                                #SET TITLE
    plt.show()

#PLOT INITIAL K-MEAN LOCATIONS AND 3D DATA
def initial_plot_3D(data_frame, k_locations, dim_dict):                 
    param_list = sorted(dim_dict.keys())
    fig = plt.figure()
    x = data_frame.loc[:, [dim_dict[param_list[0]]]]                    #X DATAPOINT POSITIONS
    y = data_frame.loc[:, [dim_dict[param_list[1]]]]                    #Y DATAPOINT POSITIONS
    z = data_frame.loc[:, [dim_dict[param_list[2]]]]                    #Z DATAPOINT POSITIONS
    x_k = k_locations[:,0]                                              #X MEAN POSITIONS
    y_k = k_locations[:,1]                                              #Y MEAN POSITIONS
    z_k = k_locations[:,2]                                              #Z MEAN POSITIONS
    ax = fig.add_subplot(111, projection='3d')                          #CREATE 3D PLOT
    ax.scatter(x,y,z, c = 'grey')                                       #PLOT DATAPOINTS IN 3D USING GREY DATAPOINTS
    ax.scatter(x_k,y_k,z_k, c='k', marker= '*')                         #PLOT MEANS IN 3D USING BLACK STARS
    ax.set_xlabel([dim_dict[param_list[0]]])                            #SET X-AXIS LABEL
    ax.set_ylabel([dim_dict[param_list[1]]])                            #SET Y-AXIS LABEL
    ax.set_zlabel([dim_dict[param_list[2]]])                            #SET Z-AXIS LABEL
    plt.title('INITIAL K-MEAN ASSIGNMENT')                              #SET TITLE
    plt.show()

#PLOT FINAL K-MEAN LOCATIONS AND 3D DATA
def final_plot_3D(data_frame, k_locations, dim_dict, k_tracker):
    color_list = ['b', 'r', 'g', 'magenta', 'cyan', 'blueviolet',\
    'orange', 'yellow', 'palegreen', 'grey', 'lime', 'peru', \
    'teal', 'hotpink', 'cornflowerblue', 'lightcoral', 'darkgray',\
    'whitesmoke', 'rosybrown', 'firebrick', 'salmon', 'chocolate',\
    'bisque', 'tan', 'gold', 'olive', 'honeydew','thistle', 'k']        #DEFINE COLOUR LIST FOR 29 CLUSTERS
    param_list = sorted(dim_dict.keys())                                
    fig = plt.figure()
    x = data_frame.loc[:, [dim_dict[param_list[0]]]]                    #X DATAPOINT POSITIONS
    y = data_frame.loc[:, [dim_dict[param_list[1]]]]                    #Y DATAPOINT POSITIONS
    z = data_frame.loc[:, [dim_dict[param_list[2]]]]                    #Z DATAPOINT POSITIONS
    x_k = k_locations[:,0]                                              #X MEAN POSITIONS
    y_k = k_locations[:,1]                                              #Y MEAN POSITIONS
    z_k = k_locations[:,2]                                              #Z MEAN POSITIONS
    ax = fig.add_subplot(111, projection='3d')                          #CREATE 3D PLOT
    for k in range(1,len(k_locations)+1):                               #CYCLE THROUGH CLUSTERS
        print("Cluster", k , color_list[k-1])                           #PRINT COLOUR KEY FOR USER (NEEDED DUE TO DIFFICULTLY IN ADDING LEGEND/CLUSTERS NUMBERS CHANGE EACH RUN)
    for i in range(0,len(data_frame)):                                  #CYCLE THROUGH DATAPOINTS
        for k in range(1,len(k_locations)+1):                           #CYCLE THROUGH CLUSTERS
            if k_tracker[i] == k:                                       #IF DATAPOINT BELONGS TO CLUSTER K ('k_tracker' IS USED TO DETERMINE THIS)
                ax.scatter(x.iat[i,0], y.iat[i,0], z.iat[i,0],\
                c= color_list[k-1])                                     #PLOT THE DATAPOINT WHICH APPROPRIATE COLOUR
    ax.scatter(x_k,y_k,z_k, c='k', marker= '*')                         #PLOT MEANS IN 3D USING BLACK STARS       
    ax.set_xlabel([dim_dict[param_list[0]]])                            #SET X-AXIS LABEL
    ax.set_ylabel([dim_dict[param_list[1]]])                            #SET X-AXIS LABEL
    ax.set_zlabel([dim_dict[param_list[2]]])                            #SET X-AXIS LABEL
    plt.title('FINAL K-MEAN ASSIGNMENT')                                #SET TITLE
    plt.show()