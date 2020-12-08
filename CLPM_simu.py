#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:48:15 2020

@author: marco
"""

import numpy  as np

######################
 ## First Scenario ##
###################### 

n_nodes = 60
n_clusters = 3
n_segments = 4
low_rate = 0.5
high_rate = 5
Lbd =  low_rate * np.ones(shape = (n_clusters, n_clusters, n_segments))
Z = np.concatenate((np.repeat(0,n_nodes/3), np.repeat(1,n_nodes/3), np.repeat(2,n_nodes/3)))
#Z =np.random.choice(np.arange(n_clusters), n_nodes, replace = True)
segment_lengths = np.array([10,10,10,10])
starts = np.cumsum(segment_lengths) - segment_lengths[0]# point in time when each phase starts
ends = np.cumsum(segment_lengths)

A = np.zeros(shape = (n_nodes, n_nodes, n_segments), dtype  = np.int)

# First segment : ER
for i in range(n_nodes-1):
    for j in range(i+1, n_nodes):
        Zi = Z[i]
        Zj = Z[j]
        A[i,j,0] = np.random.poisson(Lbd[Zi,Zj,0])
        A[j,i,0] = A[i,j,0]
        
# Second segment: Emerging communities
Lbd[:,:,1][np.diag_indices_from(Lbd[:,:,1])] = [5,5,5]
for i in  range(n_nodes-1):
    for j in range(i+1, n_nodes):       
        Zi = Z[i]
        Zj = Z[j]
        A[i,j,1] = np.random.poisson(Lbd[Zi,Zj,1])
        A[j,i,1] = A[i,j,1]
        
# Third segment: Community - splitting
Lbd[:,:,2][np.diag_indices_from(Lbd[:,:,2])] = [5,5,5]
Z[:int(n_nodes/2)] = 0
Z[int(n_nodes/2):] = 2         
for i in  range(n_nodes-1):
    for j in range(i+1, n_nodes):       
        Zi = Z[i]
        Zj = Z[j]
        A[i,j,2] = np.random.poisson(Lbd[Zi,Zj,2])
        A[j,i,2] = A[i,j,2]
        
# Fourth segment: emerging Hub (2) 
Lbd[0,0,3] =0.0
Lbd[0,1,3] = 7.5
Lbd[1,0,3] = 7.5
Z = np.repeat(1,n_nodes)
Z[5] = 0           
for i in  range(n_nodes-1):
    for j in range(i+1, n_nodes):       
        Zi = Z[i]
        Zj = Z[j]
        A[i,j,3] = np.random.poisson(Lbd[Zi,Zj,3])
        A[j,i,3] = A[i,j,3]        
        
edgelist = np.zeros(shape = (int(np.sum(A,dtype = np.int)/2), 3))
index = 0
for segment in range(len(segment_lengths)):
    for  i in range(n_nodes-1):
        for j in range(i+1, n_nodes):
            if  (A[i,j,segment]>0):
                for l in range(A[i,j,segment]):
                    edgelist[index, 0] = np.random.uniform(low = starts[segment], high = ends[segment])
                    edgelist[index, 1] = i
                    edgelist[index, 2] = j
                    index += 1


np.savetxt("input/edgelist.csv", edgelist, delimiter = ',')
np.savetxt("input/cluster_memberships_true.csv", Z, delimiter = ',')






        