#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:48:15 2020

@author: riccardo
"""

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from CLPM_fit import *
from CLPM_plot import *

device = "cpu"

folder = ''

### DATA AND PARAMETER INITIALISATION

edgelist = pd.read_csv(folder+'input/edgelist.csv')

n_changepoints = 10
time_max = 4# in this application we can use the true value
# time_max = np.max(edgelist.iloc[:,0])
changepoints = torch.tensor( np.linspace(start = 0.0, stop = time_max + 0.0001, num = n_changepoints) , dtype = torch.float64, device = device) 
np.savetxt(folder+'output/changepoints.csv', changepoints, delimiter = ',')

timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long, device = device)
n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)

Z = torch.tensor(np.random.normal(size = (n_nodes,2,(n_changepoints))), dtype = torch.float64, device = device, requires_grad = True)
beta = torch.tensor(np.random.normal(size = 1), dtype = torch.float64, device = device, requires_grad = True) # intercept  term


### OPTIMISATION
epochs = 1000
learning_rate = 2e-4
optimiser = torch.optim.SGD([{'params': beta, "lr": 1e-06},
                             {'params': Z},], 
                            lr=learning_rate)
#optimiser = torch.optim.SGD([beta,Z], lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = epochs // 10, gamma = 0.995)
loss_function_values = np.zeros(epochs)
for epoch in range(epochs):
    loss_function_values[epoch] = FitOneShot(dataset, Z, beta, optimiser).item()
    print("Epoch:", epoch, "\t\tLR (beta):", "{:2e}".format(optimiser.param_groups[0]['lr']), "\t\tLR (Z):", "{:2e}".format(optimiser.param_groups[1]['lr']), "\t\tLoss:", round(loss_function_values[epoch],3))


### PLOTS

outvid = folder + 'results/video.mp4'
frames_btw = 20
node_colors = fade_node_colors(dataset, Z, bending = 1)
node_sizes = fade_node_sizes(dataset, bending = 1)
dpi = 100
period = 1
size = (1200,900)
is_color = True
formato = 'mp4v'

clpm_animation(outvid, Z.detach().numpy(), changepoints.detach().numpy(), frames_btw, node_colors, node_sizes, dpi, period, size, is_color, formato)

#plt.plot(loss_function_values)


### EXPORT OUTPUT

Z_long_format = np.zeros([Z.shape[0]*Z.shape[1]*Z.shape[2], 4])# just writing the contents of this array in a long format so that I can read it with R
index = 0
for i in range(Z.shape[0]):
	for d in range(Z.shape[1]):
		for t in range(Z.shape[2]):
			Z_long_format[index, 0] = i+1
			Z_long_format[index, 1] = d+1
			Z_long_format[index, 2] = t+1
			Z_long_format[index, 3] = Z[i,d,t]
			index += 1

pd.DataFrame(Z_long_format).to_csv(folder+'output/positions.csv', index = False, header = False)
pd.DataFrame(loss_function_values).to_csv(folder+'output/loss_function_values.csv', index = False, header = False)




