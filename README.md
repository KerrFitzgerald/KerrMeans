My first github repository to host python code for a k-means algorithms.
Algorithms used can be found in "Information Theory, Inference & Learning Algorithms" by David J.C. Mackay

The code can be run on linux using:
python Soft_KM_V1.py casename.csv cluster_number col_name_dim1 col_name_dim2 ..... col_name_dimN interest_parameter number_of_interations normalisation_setting minimum_number_of_points_in_cluster
For example, with the suplied test data use:
2D:
python Soft_KM_V1.py test_5D.csv 10 Test_DIM1 Test_DIM2 Manufacturer 100 no 10 3
3D:
python Soft_KM_V1.py test_5D.csv 10 Test_DIM1 Test_DIM2 Test_DIM3 Manufacturer 100 no 10 3
5D:
python Soft_KM_V1.py test_5D.csv 10 Test_DIM1 Test_DIM2 Test_DIM3 Test_DIM4 Test_DIM5 Manufacturer 100 no 10 3

A number of improvements could be made to this code and repository. Examples include:
1)  Add a feature to stop 'update loop' once mean values do not move
9)  Implement Version 2 algorithm
10) Implement Version 3 algorithm
12) Change code to allow automated elbow method
13) Implement the silhouette method

Completed improvements include:
2)  Reduce the amount of print statements by using loops and a python dictionary
3)  Improve the test data so that it contains 10 clusters
4)  Add option for 2D functionality using an 'if' loop and additional input parameter
5)  Split code into functions which may also easy implementation of elbow method
6)  Copy KerrMeans3D.py and use this to create a soft k-means version
7)  Expand codes to work with more than 3 dimesnions
8)  Implement elbow method
11) Improve the test data so that it contains 10 dimensions


