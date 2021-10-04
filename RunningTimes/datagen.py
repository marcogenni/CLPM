#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:48:15 2020

@author: marco
"""

import numpy  as np

np.random.seed(12345)

#####################
  ## new version ##
#####################

tot_n_nodes = [30, 60, 90, 120, 150, 180]

for n_nodes in tot_n_nodes:
  print('number of nodes :{} '.format(n_nodes))
  n_clusters = 3
  low_rate = 0.45
  high_rate = 2
  eps = .1
  
  # reinforcing communities 
  a = np.linspace(1,high_rate+eps,40)
  len_a = len(a)
  len_a_mezzi = np.round(len(a)/2)
  # changing rates for the switching node (second part)
  x = np.concatenate((a[0:int(len_a_mezzi)], np.repeat(low_rate,(len(a)-int(len_a_mezzi)))))
  y = np.concatenate((np.repeat(low_rate,(len(a)-int(len_a_mezzi))), a[int(len_a_mezzi):]))
  ##
  n_segments = len(a) # number of time intervals!
  segment = np.arange(1,int(len(a))+1)
  
  ###
  Z = np.concatenate((np.repeat(0,n_nodes/2), np.repeat(2,n_nodes/2)))
  Z[0] = 1
  
  
  
  Lbd = np.zeros(shape=(n_clusters, n_clusters, n_segments))
  A = np.zeros(shape = (n_nodes, n_nodes, n_segments), dtype  = np.int)
  for idx in range(n_segments):
      Lbd[:,:,idx] = [[a[idx],x[idx],low_rate],[x[idx], a[idx], y[idx]],[low_rate,y[idx],a[idx]]]
      for i in  range(n_nodes-1):
          for j in range(i+1, n_nodes):       
              Zi = Z[i]
              Zj = Z[j]
              A[i,j,idx] = np.random.poisson(Lbd[Zi,Zj,idx])
              A[j,i,idx] = A[i,j,idx]
  
  edgelist = np.zeros(shape = (int(np.sum(A,dtype = np.int)/2), 3))
  index = 0
  for segment in range(n_segments):
      for  i in range(n_nodes-1):
          for j in range(i+1, n_nodes):
              if  (A[i,j,segment]>0):
                  for l in range(A[i,j,segment]):
                      edgelist[index, 0] = segment + np.random.uniform()
                      edgelist[index, 1] = i
                      edgelist[index, 2] = j
                      index += 1
  
  
  np.savetxt("edgelist_" + str(n_nodes) + ".csv", edgelist, delimiter = ',')

        
