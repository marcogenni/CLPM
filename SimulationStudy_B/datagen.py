#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:48:15 2020

@author: riccardo
"""

import math
import numpy as np

np.random.seed(12345)

n_nodes = 20
space_limit = 1
beta = 5
n_changepoints_per_phase = 20
phase_time_length = 5
time_changepoints = np.linspace(start = 0, stop = 2*phase_time_length, num = 2*n_changepoints_per_phase)
time_segment_length = time_changepoints[1] - time_changepoints[0]
distances_at_changes = np.zeros((n_nodes, n_nodes, 2*n_changepoints_per_phase))

angles_seq = np.linspace(start = 0, stop = 2*math.pi, num = n_nodes, endpoint = False)
positions_start = np.zeros([n_nodes, 2])
for i in range(n_nodes):
    positions_start[i,0] = space_limit * math.cos(angles_seq[i])
    positions_start[i,1] = space_limit * math.sin(angles_seq[i])

positions = np.zeros([n_nodes, 2, 2*n_changepoints_per_phase])
for i in range(n_nodes):
    for t in range(n_changepoints_per_phase):
        k = t / (n_changepoints_per_phase-1)
        positions[i,:,t] = (1-k) * positions_start[i,:]
        positions[i,:,n_changepoints_per_phase+t] = k * positions_start[i,:]

distances = np.zeros([n_nodes, n_nodes, 2*n_changepoints_per_phase])
for i in range(n_nodes):
    for j in range(n_nodes):
        if (i < j):
            for t in range(2*n_changepoints_per_phase):
                distances[i,j,t] = sum((positions[i,:,t]-positions[j,:,t])**2)

n_events_per_interval = np.zeros([n_nodes, n_nodes, 2*n_changepoints_per_phase-1])
for i in range(n_nodes):
    for j in range(n_nodes):
        if (i < j):
            for t in range(2*n_changepoints_per_phase-1):
                n_events_per_interval[i,j,t] = np.random.poisson(time_segment_length * math.exp(beta - distances[i,j,t]))

edgelist = np.zeros([n_events_per_interval.sum().astype(int),3])
index = 0
for i in range(n_nodes):
    for j in range(n_nodes):
        if (i < j):
            for t in range(2*n_changepoints_per_phase-1):
                if (n_events_per_interval[i,j,t] > 0):
                    for event in range(n_events_per_interval[i,j,t].astype(int)):
                        edgelist[index, 0] = np.random.uniform(low = time_changepoints[t], high = time_changepoints[t+1])
                        edgelist[index, 1] = i
                        edgelist[index, 2] = j
                        index += 1

np.savetxt("./edgelist.csv", edgelist, delimiter = ',')


