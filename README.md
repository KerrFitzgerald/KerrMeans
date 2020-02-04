# KerrMeans
My first github repository to host python code for a hard k-means algorithm.
The code can be run on linux using:
python KerrMeans3D.py casename.csv cluster_number col_name_x col_name_y col_name_z interest_parameter number_of_interations minimum_number_of_points_in_cluster
For example, with the suplied test data use:
python KerrMeans3D.py test_3D.csv 5 Test_X Test_Y Test_Z Manufacturer 1000 2

A number of improvements could be made to this code and repository. Examples include:
1) Add a feature to stop 'update loop' once mean values do not move
2) Reduce the amount of print statements by using loops and a python dictionary
3) Improve the test data so that it contains 10 clusters
4) Add option for 2D functionality using an 'if' loop and additional input parameter
5) Split code into functions which may also easy implementation of elbow method
5) Copy KerrMeans3D.py and use this to create a soft k-means version
6) Expand codes to work with more than 3 dimesnions


