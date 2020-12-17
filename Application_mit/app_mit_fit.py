#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  17 00:01:15 2020

@author: riccardo
"""

import math
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../')
from CLPM_fit import MDataset, FitOneShot
from CLPM_plot import make_video

device = "cpu"


### DATA AND PARAMETER INITIALISATION

edgelist = pd.read_csv('input/edgelist.csv')

n_changepoints = 50
time_max = 10# in this application we know the true value
# time_max = np.max(edgelist.iloc[:,0])
changepoints = torch.tensor( np.linspace(start = 0.0, stop = time_max + 0.0001, num = n_changepoints) , dtype = torch.float64, device = device) 
np.savetxt("output/changepoints.csv", changepoints, delimiter = ',')

timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long, device = device)
n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)

Z = torch.tensor(np.random.normal(size = (n_nodes,2,(n_changepoints))), dtype = torch.float64, device = device, requires_grad = True)
beta = torch.tensor(np.random.normal(size = 1), dtype = torch.float64, device = device, requires_grad = True)


### OPTIMISATION

### OPTIMISATION
epochs = 1000
learning_rate = 1e-3
optimiser = torch.optim.SGD([{'params': beta, "lr": 2e-06},
                             {'params': Z},], 
                            lr=learning_rate)
#optimiser = torch.optim.SGD([beta,Z], lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = epochs // 10, gamma = 0.995)
loss_function_values = np.zeros(epochs)
for epoch in range(epochs):
    loss_function_values[epoch] = FitOneShot(dataset, Z, beta, optimiser).item()
    print("Epoch:", epoch, "\t\tLR (beta):", "{:2e}".format(optimiser.param_groups[0]['lr']), "\t\tLR (Z):", "{:2e}".format(optimiser.param_groups[1]['lr']), "\t\tLoss:", round(loss_function_values[epoch],3))


### EXPORT OUTPUT

path = ''

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

pd.DataFrame(Z_long_format).to_csv(path+'output/positions.csv', index = False, header = False)
pd.DataFrame(loss_function_values).to_csv(path+'output/loss_function_values.csv', index = False, header = False)


####################
 ## Plot results ##
####################

import matplotlib.pyplot as plt

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

