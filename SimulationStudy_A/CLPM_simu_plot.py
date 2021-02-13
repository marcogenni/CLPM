#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:55:22 2021

@author: marco
"""

import pandas as pd
import numpy as np
import torch

device = 'cpu'

import sys
sys.path.append('../')
from CLPM_fit import *
from CLPM_plot import *
folder = ''

edgelist = pd.read_csv('input/edgelist.csv')
changepoints = np.loadtxt("input/changepoints.csv", delimiter = ',')
n_changepoints = len(changepoints)
timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long, device = device)
n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)

Z = torch.zeros(size = (n_nodes,2,(n_changepoints)), dtype = torch.float64)
Z_long = pd.read_csv(folder+'output/positions.csv', header = None)
for row in range(len(Z_long)):
    i = Z_long.iloc[row,0]-1
    d = Z_long.iloc[row,1]-1
    t = Z_long.iloc[row,2]-1
    val = Z_long.iloc[row,3]
    Z[i.astype('int'),d.astype('int'),t.astype('int')] = val

Z = torch.abs(Z)

# times
from datetime import datetime


outvid = folder + 'results/video.mp4'
frames_btw = 10
node_colors = fade_node_colors(dataset, Z, bending = 1)
node_sizes = fade_node_sizes(dataset, bending = 1)
dpi = 250
period = 1
size = (1200,900)
is_color = True
formato = 'mp4v'

clpm_animation(outvid, 
               Z.detach().numpy(), 
               changepoints, 
               frames_btw, 
               node_colors, 
               node_sizes,
               dpi, period, size, is_color, formato)

#plt.plot(loss_function_values)



