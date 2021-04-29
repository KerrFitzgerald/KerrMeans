# Program for Hard K-means clustering of N-dimensional data.

import pandas as pd
import sys
import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def dtframe_read(datafile):
    """Read a .csv file into a Pandas dataframe

    Parameters:
       datafile   (str): Name of file (must be .csv format)

    Returns:
       dt_frame (pd df): Pandas dataframe
    """

    dt_frame = pd.read_csv(datafile, delimiter=',')

    return dt_frame


def z_score_norm(dt_frame, interest):
    """Apply Z-score normalization to selected dataframe columns

    Parameters:
       dt_frame (pd df): Pandas dataframe
       interest  (list): Column names (dimensions) for K-means algorithm

    Returns:
       sd_frame (pd df): Pandas dataframe with normalized interest columns
    """

    sd_frame = dt_frame.copy()
    for i in range(len(interest)):
        sd_frame[interest[i]] = ((sd_frame[interest[i]] -
                                 sd_frame[interest[i]].mean()) /
                                 sd_frame[interest[i]].std())

    return sd_frame


def k_pos_assign(dt_frame, interest, k_points):
    """Assign k centroids to initial random positions

    Parameters:
       dt_frame (pd df): Pandas dataframe
       interest  (list): Column names (dimensions) for K-means algorithm
       k_points   (int): Number of centroids

    Returns:
       k_positn (np ar): Randomized centroid starting positions
    """

    # Number of clusters
    k_points = int(k_points)
    # Create array with 'k_points' columns & 'len(interest)' rows
    k_positn = np.zeros((int(k_points), len(interest)))
    # Cycle through dataframe columns to determine min & max interest values
    for i in range(0, len(interest)):
        min_inter = (dt_frame[interest[i]].min())
        max_inter = (dt_frame[interest[i]].max())
        for k in range(0, int(k_points)):
            # Assign dimension position to random number between min & max
            k_positn[k, i] = np.random.uniform(low=min_inter,
                                               high=max_inter)

    return k_positn


def euclidn_dist(dt_frame, interest, k_positn):
    """Calculate euclidean distance between centroids and datapoints

    Parameters:
       dt_frame (pd df): Pandas dataframe
       interest  (list): Column names (dimensions) for K-means algorithm
       k_positn (np ar): Position of centroids

    Returns:
       euc_dist (np ar): Array of data point distance to each cluster
                         has shape (num_points, k_centroid)
    """

    t_points = int(len(dt_frame))
    k_points = int(len(k_positn))
    euc_dist = np.zeros((t_points, k_points))
    # Vectorised implementation of euclidean distance calculation
    for k in range(0, k_points):
        point = 0
        for j in range(0, len(interest)):
            point += (dt_frame[interest[j]] - k_positn[k, j]) ** 2
        euc_dist[:, k] = np.sqrt(point)

    return euc_dist


def hard_k_track(euc_dist):
    """Track which datapoint belongs to a cluster (hard assignment)

    Parameters:
       euc_dist (np ar): Array of data point distance to each cluster
                         has shape (num_points, k_centroid)

    Returns:
       hrd_trck (np ar): Binary array of data point centroid assignment
                         has shape (num_points, k_centroid)
    """

    ar_shape = (euc_dist.shape)
    hrd_trck = np.zeros((ar_shape))
    for i in range(0, ar_shape[0]):
        result = np.where(euc_dist[i, :] == np.amin(euc_dist[i, :]))
        hrd_trck[i, result[0]] = np.int(1)

    return hrd_trck


def hard_updater(dt_frame, hrd_trck, interest):
    """Track which datapoint belongs to a cluster (hard assignment)

    Parameters:
       dt_frame (pd df): Pandas dataframe
       interest  (list): Column names (dimensions) for K-means algorithm
       hrd_trck (np ar): Binary array of data point centroid assignment
                         has shape (num_points, k_centroid)


    Returns:
       u_positn (np ar): Updated centroid positions for hard K-means
    """

    ar_shape = (hrd_trck.shape)
    u_positn = np.zeros((ar_shape[1], len(interest)))
    for k in range(0, ar_shape[1]):
        temp_res = hrd_trck[:, k]
        temp_sum = np.sum(temp_res)
        for j in range(0, len(interest)):
            temp_dim = dt_frame[interest[j]].to_numpy()
            u_positn[k, j] = np.sum(temp_dim * temp_res) / temp_sum

    return u_positn


def conv_checker(n_positn, o_positn):
    """Determine distance between old & new centroid posiions
       (used to check for algorithm convergence)

    Parameters:
       n_positn (np ar): New centroid positions
       o_positn (np ar): Old centroid positions

    Returns:
       distchck (np ar): Distance between old  new centroids
    """

    distchck = np.sum(((n_positn - o_positn)**2), axis=1)

    return distchck


def hrd_sqd_dist(dt_frame, k_positn, interest, hrd_trck):
    """Determine total sum of squared distance for each
       centroid and its assigned datapoints

    Parameters:
       dt_frame (pd df): Pandas dataframe
       k_positn (np ar): Position of centroids
       interest  (list): Column names (dimensions) for K-means algorithm
       hrd_trck (np ar): Binary array of data point centroid assignment
                         has shape (num_points, k_centroid)
    Returns:
       col_sums (np ar): Total sum of squared distance for clusters & assigned
                         datapoints
    """

    dist_arr = euclidn_dist(dt_frame, interest, k_positn)
    sqd_dist = (np.multiply(dist_arr, hrd_trck))**2
    col_sums = np.sum(sqd_dist, axis=0)
    ssd_totl = np.sum(col_sums)

    return col_sums


def hard_k_means(
                 datafile, interest, max_iter, k_points, min_pnts, norm_flg,
                 threshld
                ):
    """Implement the hard K-means algorithm

    Parameters:
       datafile   (str): Name of file (must be .csv format)
       interest  (list): Column names (dimensions) for K-means algorithm
       max_iter   (int): Maximum number of iterations
       k_points   (int): Number of centroids
       min_pnts   (int): Minimum number of points allowed in a cluster
       norm_flg  (bool): Normalization flag 'True' or 'False'
       threshld   (flt): Convergence criteria

    Returns:
       dt_frame (pd df): Pandas dataframe (with interest columns normalized)
       n_positn (np ar): Final centroid positions
       hrd_trck (np ar): Final binary array of data point centroid assignment
                         has shape (num_points, k_centroid)
       ssd_htot   (flt): Total sum of squared distance for clusters & assigned
                         datapoints
    """

    dt_frame = dtframe_read(datafile)
    # Normalize dataframe interest columns if requested
    if norm_flg is True:
        dt_frame = z_score_norm(dt_frame, interest)
    # Assign initial centroid positions
    k_positn = k_pos_assign(dt_frame, interest, k_points)
    n_positn = k_positn
    for i in range(0, max_iter):
        o_positn = n_positn
        # Calculate distance between datapoints and clusters
        euc_dist = euclidn_dist(dt_frame, interest, o_positn)
        # Hard assign datapoints to clusters
        hrd_trck = hard_k_track(euc_dist)
        # Determine number of points in each cluster
        clst_pts = np.sum(hrd_trck, axis=0)
        # Check to see that the clusters contain minimum number of points
        for k in range(0, len(clst_pts)):
            if clst_pts[k] <= min_pnts:
                # Randomly re-initialise clusters
                n_positn = k_pos_assign(dt_frame, interest, k_points)
            else:
                # Update the position of the centroids
                n_positn = hard_updater(dt_frame, hrd_trck, interest)
        # Check to see if the algorithm has converged
        ctrd_chn = conv_checker(n_positn, o_positn)
        if np.sum(ctrd_chn) <= threshld:
            break
    # Calculate total sum of square distance for clusters & assigned points
    ssd_htot = np.sum(hrd_sqd_dist(dt_frame, n_positn, interest, hrd_trck))

    return n_positn, hrd_trck, clst_pts, ssd_htot


def stat_hrd_run(
                 datafile, interest, max_iter, min_pnts, norm_flg,
                 threshld, seed_lst, max_pnts
                ):
    """Run the hard K-means algorithm for each number of centroids
       up to the maximum number of centroids specified by the user.
       For each number of centroids conduct multiple runs using a
       list of random number seeds specified by the user & choose
       the result with the smallest sum of squared distance value.

    Parameters:
       datafile   (str): Name of file (must be .csv format)
       interest  (list): Column names (dimensions) for K-means algorithm
       max_iter   (int): Maximum number of iterations
       min_pnts   (int): Minimum number of points allowed in a cluster
       norm_flg  (bool): Normalization flag 'True' or 'False'
       threshld   (flt): Convergence criteria
       seed_lst  (list): List of random number seeds
       max_pnts   (int): Maximum number of clusters to try

    Returns:
       rsltdict  (dict): Dictionary containing run paramaters & results

    """

    ssd_vs_k = np.zeros(max_pnts+1)
    top_seed = np.zeros(max_pnts+1)
    rsltdict = {}
    for k in range(1, max_pnts+1):
        rsltdict["Result_k{0}".format(k)] = None
    for k in range(1, max_pnts+1):
        min_SSD = 1e16
        fnl_pnts = None
        fnl_clst = None
        fnl_trck = None
        for i in range(0, len(seed_lst)):
            np.random.seed(seed_lst[i])
            f_positns, hrd_trck, clst_pnts, ssd_htot = hard_k_means(datafile,
                                                                    interest,
                                                                    max_iter,
                                                                    k,
                                                                    min_pnts,
                                                                    norm_flg,
                                                                    threshld
                                                                    )
            if ssd_htot < min_SSD:
                min_SSD = ssd_htot
                ssd_vs_k[k] = min_SSD
                top_seed[k] = seed_lst[i]
                fnl_clst = clst_pnts
                fnl_trck = hrd_trck
                fnl_pnts = f_positns
        rsltdict["Result_k{0}".format(k)] = [len(interest),
                                             interest,
                                             k,
                                             top_seed[k],
                                             ssd_vs_k[k],
                                             fnl_clst,
                                             fnl_trck,
                                             fnl_pnts
                                             ]
    dt_frame = dtframe_read(datafile)
    if norm_flg is True:
        dt_frame = z_score_norm(dt_frame, interest)
    return dt_frame, rsltdict


def elbow_method(rsltdict):
    """Plot the 'sum of the squared distance' vs 'number of clusters'
       to allow the user to implement the elbow method in order to choose
       an appropriate number of clusters. Appropriate number of clusters
       will be somewhere around the inflection point.

    Parameters:
       rsltdict  (dict): Dictionary containing run paramaters & results

    Returns:
       None

    """

    centroid = []
    ssd_htot = []
    for key in rsltdict.keys():
        centroid.append(rsltdict[key][2])
        ssd_htot.append(rsltdict[key][4])
    fig = plt.figure()
    plt.plot(centroid, ssd_htot)
    plt.title('ELBOW PLOT')
    plt.xlabel('Number of Centroids')
    plt.ylabel('Sum of Squared Distance')
    plt.show()

    return None


def final_plot_2D(dt_frame, rsltdict):
    """Plot the 'centroid co-ordinates' and 'original data points' in 2D.
       Data-points will be colour coded corresponding to the centroid
       they belong to.

    Parameters:
       dt_frame (pd df): Pandas dataframe
       rsltdict  (dict): Dictionary containing run paramaters & results

    Returns:
       None

    """

    color_list = ['b', 'r', 'g', 'magenta', 'cyan', 'blueviolet',
                  'orange', 'yellow', 'palegreen', 'grey', 'lime',
                  'peru', 'teal', 'hotpink', 'cornflowerblue',
                  'lightcoral', 'darkgray', 'whitesmoke', 'rosybrown',
                  'firebrick', 'salmon', 'chocolate', 'bisque', 'tan',
                  'gold', 'olive', 'honeydew', 'thistle', 'k']

    for key in rsltdict.keys():
        interest = rsltdict[key][1]
        fnl_trck = rsltdict[key][6]
        fnl_pnts = rsltdict[key][7]
        fig = plt.figure()
        x = dt_frame[interest[0]]
        y = dt_frame[interest[1]]
        x_k = fnl_pnts[:, 0]
        y_k = fnl_pnts[:, 1]
        for i in range(0, len(dt_frame)):
            for k in range(0, len(fnl_pnts)):
                if fnl_trck[i, k] == 1:
                    plt.scatter(x[i], y[i], c=color_list[k])
        plt.scatter(x_k, y_k, c='k', marker='*')
        plt.xlabel(interest[0])
        plt.ylabel(interest[1])
        plt.title('FINAL K-MEAN ASSIGNMENT: K = ' + str(k+1))
        plt.show()

    return None


def final_plot_3D(dt_frame, rsltdict):
    """Plot the 'centroid co-ordinates' and 'original data points' in 3D.
       Data-points will be colour coded corresponding to the centroid
       they belong to.

    Parameters:
       dt_frame (pd df): Pandas dataframe
       rsltdict  (dict): Dictionary containing run paramaters & results

    Returns:
       None

    """

    color_list = ['b', 'r', 'g', 'magenta', 'cyan', 'blueviolet',
                  'orange', 'yellow', 'palegreen', 'grey', 'lime',
                  'peru', 'teal', 'hotpink', 'cornflowerblue',
                  'lightcoral', 'darkgray', 'whitesmoke', 'rosybrown',
                  'firebrick', 'salmon', 'chocolate', 'bisque', 'tan',
                  'gold', 'olive', 'honeydew', 'thistle', 'k']

    for key in rsltdict.keys():
        interest = rsltdict[key][1]
        fnl_trck = rsltdict[key][6]
        fnl_pnts = rsltdict[key][7]
        fig = plt.figure()
        x = dt_frame[interest[0]]
        y = dt_frame[interest[1]]
        z = dt_frame[interest[2]]
        ax = fig.add_subplot(111, projection='3d')
        x_k = fnl_pnts[:, 0]
        y_k = fnl_pnts[:, 1]
        z_k = fnl_pnts[:, 2]
        for i in range(0, len(dt_frame)):
            for k in range(0, len(fnl_pnts)):
                if fnl_trck[i, k] == 1:
                    ax.scatter(x[i], y[i], z[i], c=color_list[k])
        ax.scatter(x_k, y_k, z_k, c='k', marker='*')
        ax.set_xlabel(interest[0])
        ax.set_ylabel(interest[1])
        ax.set_zlabel(interest[2])
        plt.title('FINAL K-MEAN ASSIGNMENT: K = ' + str(k+1))
        plt.show()

    return None


def summary_plots(dt_frame, rsltdict):
    """Cycle through the results dictionary generated using the stat_hrd_run
       function and use the final_plot_2D or final_plot_3D functions to create
       plots for each number of centroids tested. If less than 2 or more than 3
       dimensions are selected plots cannot be generated.

    Parameters:
       dt_frame (pd df): Pandas dataframe
       rsltdict  (dict): Dictionary containing run paramaters & results

    Returns:
       None

    """

    for key in rsltdict.keys():
        tot = 0
        if rsltdict[key][0] == 2:
            tot += 2
        if rsltdict[key][0] == 3:
            tot += 3
    if tot % (len(rsltdict.keys())) == 2:
        print('Generating 2D plots...')
        final_plot_2D(dt_frame, rsltdict)
    if tot % (len(rsltdict.keys())) == 3:
        print('Generating 3D plots...')
        final_plot_3D(dt_frame, rsltdict)
    if tot % (len(rsltdict.keys())) != 2 and tot % (len(rsltdict.keys())) != 3:
        print('Result not suitable for plotting')

    return None


def results_print(rsltdict):
    """Cycle through the results dictionary generated using the stat_hrd_run
       function and use the final_plot_2D or final_plot_3D functions to create
       plots for each number of centroids tested. If less than 2 or more than 3
       dimensions are selected plots cannot be generated.

    Parameters:
       rsltdict  (dict): Dictionary containing run paramaters & results

    Returns:
       None

    """

    star = '*'*100
    print('#########################')
    print('#####RESULTS SUMMARY#####')
    print('#########################')
    print(star)
    for key in rsltdict.keys():
        print('Number of parameters:          ', rsltdict[key][0])
        print('Parameters used:               ', rsltdict[key][1])
        break
    print(star)
    for key in rsltdict.keys():
        print('For number of centroids:       ', rsltdict[key][2])
        print('Suitable random seed is:       ', rsltdict[key][3])
        print('Min sum of squared distance:   ', rsltdict[key][4])
        print('No. of points in each cluster: ', rsltdict[key][5])
        print('Final centroid positions:    \n', rsltdict[key][7])
        print(star)

    return None

# Testing using 'wine.data'
file = 'wine.data'
col_names_all = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash',
                 'Magnesium', 'Total_phenols', 'Flavanoids',
                 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity',
                 'Hue', 'D280OD315_of_diluted_wines', 'Proline'
                 ]
col_names_2D = ['Alcohol', 'Malic_acid']
col_names_3D = ['Alcohol', 'Malic_acid', 'Alcalinity_of_ash']
col_names_4D = ['Alcohol', 'Malic_acid', 'Alcalinity_of_ash', 'Proline']

seed_list = [5, 7, 9, 12, 14, 11, 18, 21, 42,
             69, 111, 359, 709, 777, 1701, 8472
             ]

data_frame, data_results = stat_hrd_run(
                                        file,
                                        col_names_3D,
                                        500,
                                        1,
                                        True,
                                        1e-08,
                                        seed_list,
                                        15
                                        )
summary_plots(data_frame, data_results)
elbow_method(data_results)
results_print(data_results)
