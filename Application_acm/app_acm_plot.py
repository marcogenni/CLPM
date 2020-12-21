#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  21 00:01:15 2020

@author: riccardo
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from CLPM_plot import make_video

folder = ''

Z_long = np.genfromtxt(folder+'output/positions.csv', delimiter = ",")
n_nodes = np.max(Z_long[:,0]).astype(int)
n_dimensions = np.max(Z_long[:,1]).astype(int)
n_time_frames = np.max(Z_long[:,2]).astype(int)
Z = np.zeros((n_nodes, n_dimensions, n_time_frames))
for index in range(len(Z_long)):
    i_ = (Z_long[index,0]-1).astype(int)
    j_ = (Z_long[index,1]-1).astype(int)
    k_ = (Z_long[index,2]-1).astype(int)
    Z[i_,j_,k_] = Z_long[index,3]

for snap in range(Z.shape[2]):
        plt.figure("Latent Positions")
        plt.xlim((-10.0,10.0))
        plt.ylim((-10.0,10.0))
        for idi in range(n_nodes):    
            plt.plot(Z[idi,0,snap], Z[idi,1,snap], 'ro')
        plt.savefig(folder+'results/snaps/snap_'+str(snap)+'.png', dpi = 200)
        plt.close()

img=[]
for i in range(n_time_frames):
    img.append(folder+'results/snaps/snap_'+str(i)+'.png')

test = make_video(folder+'results/video.mp4', images = img)

plt.plot(loss_function_values)

