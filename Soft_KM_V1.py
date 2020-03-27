#PYTHON PROGRAM FOR SOFT K-MEANS (VERSION 1) CLUSTERING OF N-DIMENSIONAL DATA.
#PROGRAM IS BASED ON ALGORITHM 20.7 FROM "INFORMATION THEORY, INFERENCE, & LEARNING ALGORITHMS" BY DAVID J.C. MACKAY
#CREATED BY KERR FITZGERALD, THOMAS BENNETT AND AIDEN PEAKMAN 2020.

#IMPORT RELEVANT PYTHON MODULES
import pandas as pd
import sys
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# IMPORT K-MEANS CLUSTERING FUNCTIONS
import K_Means_Functions as KMF

#IMPLEMENT SOFT K MEANS VERSION 1 ALGORITHM
def soft_k_mean_V1():
    filename, k_points, dims, interest, dim_dict, N_steps, norm, \
    beta, min_assign = KMF.input_read()                                 #READ USER INPUT USING KMF FUCNTION
    print(filename, k_points, dims, interest, dim_dict, N_steps, \
    norm, beta)                                                         #PRINT USER INPUT
    param_list = sorted(dim_dict.keys())
    if norm == 'yes':                                                   #IF USER INPUT FOR NORMALISATION SET TO 'yes'
        print('************NORMALISED DATAFRAME (COLUMNS)*************')
        soft_k_data = KMF.df_read(filename)                             #READ DATAFRAME FROM FILE
        soft_k_data = KMF.df_normalise(soft_k_data, dim_dict)           #NORMALISE COLUMNS WITH SELECTED PARAMETERS USING KMF FUNCTION
        print(soft_k_data)                                              #PRINT NORMALISED DATAFRAME
    if norm == 'no':                                                    #IF USER INPUT FOR NORMALISATION SET TO 'no'
        print('***********************DATAFRAME**********************')
        soft_k_data = KMF.df_read(filename)                             #READ DATAFRAME FROM FILE
        print(soft_k_data)                                              #PRINT NORMALISED DATAFRAME
    print('**************ORIGINAL K MEAN CO-ORDINATES************')
    k_loc = KMF.k_assign(int(dims), k_points, soft_k_data, dim_dict)    #CREATE INITIAL K-MEAN CO-ORDINATES USING KMF FUCNTION
    print(k_loc)                                                        #PRINT INITIAL K CLUSTER CO-ORDINATES
    if int(dims) == 2:                                                  #IF 2D IS BEING USED
        KMF.initial_plot_2D(soft_k_data, k_loc, dim_dict)               #PLOT DATAPOINTS AND INITIAL K-MEAN CO-ORDINATES IN 2D USING KMF FUNCTION
    if int(dims) == 3:                                                  #IF 3D IS BEING USED
        KMF.initial_plot_3D(soft_k_data, k_loc, dim_dict)               #PLOT DATAPOINTS AND INITIAL K-MEAN CO_ORDINATES IN 3D USING KMF FUNCTION
    rk_Dim_Dict={}                                                      #CREATE DICTIONARY TO ALLOW ACCESS TO DIMENSIONS OF RESPONSIBILITY ARRAY
    rkTotalDict ={}                                                     #CREATE DICTIONARY TO ALLOW TOTAL OF EACH DIMENSION OF RESPONSIBILITY ARRAY TO BE TRACKED
    for j in range(1,dims+1):                                           #CYCLE THROUGH DIMENSIONS
        rk_Dim_Dict["rk{0}".format(j)] = 0                              #CREATE DICTIONARY KEYS FOR 'DIMENSIONS OF RESPONSIBILITY ARRAY'
        rkTotalDict["rkTotal{0}".format(j)] = 0                         #CREATE DICTIONARY KEYS FOR 'TOTAL OF EACH DIMENSION OF RESPONSIBILITY ARRAY'
    rk_Dim_List = sorted(rk_Dim_Dict.keys())
    rkTotalList = sorted(rkTotalDict.keys())
    k_assign = np.zeros(int(k_points))                                  #ARRAY TO TRACK NUMBER OF POINTS ASSIGNED TO EACH CLUSTER TO ALLOW PERTURBATION
    for n in range(0,int(N_steps)):                                     #CYCLE THROUGH ITERATION (USER DEFINED MAX ITERATIONS USED)
        k_assign[:] = 0                                                 #RESET ASSIGNMENT ARRAY OF NUMBER OF POINTS ASSIGNED TO EACH CLUSTER TO ALLOW PERTURBATION
        new_dist_arr =\
        KMF.eucl_dist_arr(dims, k_loc, soft_k_data, dim_dict)           #UPDATE DISTANCE ARRAY USING KMF FUNCTION
        k_res_data = KMF.k_respons_arr(beta,new_dist_arr, k_loc)        #UPDATE K-RESPOSIBILTY ARRAY USING KMF FUNCTION
        tot_k_respons = k_res_data.sum(axis=0)                          #FIND COLUMN TOTALS (CLUSTER TOTAL) OF RESPONSIBILITY ARRAY
        for k in range(0, int(k_points)):                               #CYCLE THROUGH CLUSTERS
            for j in range(0,dims):                                     #CYCLE THROUGH DIMENSIONS
                for i in range(0, len(soft_k_data)):                    #CYCLE THROUGH DATAPOINTS
                    rk_Dim_Dict[rk_Dim_List[j]] = k_res_data[i,k] *\
                    soft_k_data[dim_dict[param_list[j]]][i]             #MULTIPLY EACH DATAPOINT CLUSTER RESPONSIBILITY BY THE DATAPOINT (J DIMENSION)
                    rkTotalDict[rkTotalList[j]] = \
                    rkTotalDict[rkTotalList[j]] + \
                    rk_Dim_Dict[rk_Dim_List[j]]                         #CALCULATE SUM OF ALL CLUSTER RESPONSIBILITY X DATAPOINTS
                    k_loc[k,j] = rkTotalDict[rkTotalList[j]]/\
                    tot_k_respons[k]                                    #UPDATE CLUSTER MEAN LOCATION BY DIVIDING BY TOTAL RESPONSIBILITY 
                rkTotalDict[rkTotalList[j]] = 0                         #RESET VALUE
        k_res_track = KMF.k_tracker(k_res_data, k_loc)                  #UPDATE RESPONSIBILITY TRACKER
        for k in range(0, int(k_points)):                               #CYCLE THROUGH CLUSTERS
            for i in range(0, len(soft_k_data)):                        #CYCLE THROUGH DATAPOINTS
                if k_res_track[i] == k+1:                               #IF RESPONSIBILITY TRACKER EQUALS CLUSTER NUMBER
                    k_assign[k] = k_assign[k] + 1                       #UPDATE ARRAY CONTAINING NUMBER OF POINTS ASSIGNED TO EACH CLUSTER
        for k in range(0, int(k_points)):                               #CYCLE THROUGH CLUSTERS
            if k_assign[k] <= min_assign:                               #IF NUMBER OF DATAPOINTS ASSIGNED TO CLUSTER IS EQUAL/LOWER THAN USER INPUT
                print("Perturbed iteration", '{0: <3}'.format(n),\
                "cluster",'{0: <3}'.format(k+1))                        #NOTIFY CLUSTER MEAN IS BEING PERTURBED
                for j in range(0,dims):                                 #CYCLE THROUGH DIMENSIONS
                    mini = (soft_k_data[dim_dict[param_list[j]]].min()) #FIND MINIMUM VALUE IN DATAFRAME COLUMN (DIMENSION)
                    maxi = (soft_k_data[dim_dict[param_list[j]]].max()) #FIND MAXIMUM VALUE IN DATAFRAME COLUMN (DIMENSION)
                    k_loc[k,j] = np.random.uniform(low=mini, high=maxi) #UPDATE CLUSTER MEAN SO IT MOVES TO NEW RANDOM POSITION
    print('*******************ELBOW METHOD SCORE***************')
    sesk_Dict={}                                                        #CREATE DICTIONARY TO STORE STORE SUM ERRORS SQUARED FOR EACH CLUSTER K
    for k in range(1, int(k_points)+1):                                 #CYCLE THROUGH CLUSTERS
        sesk_Dict["sesk{0}".format(k)] = 0                              #FORMAT DICTIONARY KEYS
        sesk_List = sorted(sesk_Dict.keys())                            #SUM ERRORS SQUARED FOR EACH CLUSTER K
    for i in range(0, len(soft_k_data)):                                #CYCLE THROUGH DATAPOINTS
        for k in range(0, int(k_points)):                               #CYCLE THROUGH CLUSTERS
            if k_res_track[i] == k+1:                                   #IF RESPONSIBILITY TRACKER EQUALS CLUSTER NUMBER
                sesk_Dict[sesk_List[k]] = sesk_Dict[sesk_List[k]]+\
                (new_dist_arr[i,k])**2                                  #SELECT DISTANCE FROM DATA POINT TO ASSIGNED CLUSTER AND SQUARE
    sesk_total = sum(sesk_Dict.values())                                #CALCULATE TOTAL OF SUM ERRORS SQUARED FOR EACH CLUSTER K
    print("Elbow Method Sum of Errors Squared Score=", sesk_total)      #PRINT SUM ERRORS SQUARED FOR USER
    print('***************FINAL K MEAN CO-ORDINATES************')
    print(k_loc)                                                        #PRINT FINAL K MEANS CO-ORDINATES
    k_res_track = KMF.k_tracker(k_res_data, k_loc)                      #DETERMINE FINAL RESPONSIBILTY TRACKER USING KMF FUNCTION
    if int(dims) == 2:                                                  #IF 2D IS BEING USED
        KMF.final_plot_2D(soft_k_data, k_loc, dim_dict, k_res_track)    #PLOT DATAPOINTS AND FINAL K-MEAN CO_ORDINATES IN 2D USING KMF FUNCTION
    if int(dims) == 3:                                                  #IF 3D IS BEING USED
        KMF.final_plot_3D(soft_k_data, k_loc, dim_dict, k_res_track)    #PLOT DATAPOINTS AND FINAL K-MEAN CO_ORDINATES IN 3D USING KMF FUNCTION
    print('**SUMMARY OF CLUSTER & K-RESPONSIBILITY INFORMATION**')
    interest_array =  soft_k_data.loc[:, interest]                      #
    unique, counts = np.unique(interest_array, return_counts=True)      #
    interest_total = len(dict(zip(unique, counts)))                     #
    clust_inter_arr = \
    pd.DataFrame(np.zeros((int(k_points),interest_total)),\
    columns = range(1,interest_total+1))                                #
    unique, counts = np.unique(k_res_track, return_counts=True)         #
    for i,x in enumerate(k_res_track):                                  #
        j = int(k_res_track[i]-1)
        print('Index', '{0: <3}'.format(i), 'belongs to cluster:',\
        '{0: <3}'.format(int(x)) , 'and has', interest,\
        '{0: <3}'.format(interest_array[i]), \
        "with score ",k_res_data[i,j])                                  #
        clust_inter_arr.iloc[int(x-1), int(interest_array[i]-1)] += 1   #
    print('')
    for j in range(len(unique)):                                        #
        print("Cluster", '{0: <2}'.format(int(unique[j])), \
        "contains:", counts[j])                                         #
    print('')
    print('**********ARRAY OF CLUSTER & INTEREST DATA***********')      #
    print('NOTE: Rows are clusters & columns are interest parameters')  #
    print(clust_inter_arr)                                              #
      
if __name__ == "__main__":
    soft_k_mean_V1()