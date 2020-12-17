#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 19:05:39 2020

@author: marco
"""
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append('../')
from CLPM_fit import MDataset, FitOneShot

device = "cpu"


#### DATA AND PARAMETER INITIALISATION

path = ""
my_step = 1/20
edgelist = pd.read_csv(path + 'input/edgelist.csv')
time_max = np.max(edgelist.iloc[:,0])
changepoints = np.arange(start = 0.0, stop = 1.0 + my_step ,  step = my_step)*(time_max + 0.0001)
# L'ultimo change point deve essere più alto del più alto interaction time
changepoints = torch.tensor(changepoints, dtype = torch.float64, device = device) 
changepoints.reshape(-1,1)
np.savetxt(path + "input/changepoints.csv", changepoints, delimiter = ',')
#changepoints = torch.tensor(pd.read_csv('input/changepoints.csv', header = None).values, dtype = torch.float64, device = device)
n_changepoints = len(changepoints)
timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long, device = device)
n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)
Z = torch.tensor(np.random.normal(size = (n_nodes,2,(n_changepoints))), dtype = torch.float64, device = device, requires_grad = True)
beta = torch.tensor(np.random.normal(size = 1), dtype = torch.float64, device = device, requires_grad = True) # intercept  term



### OPTIMISATION
epochs = 3000
learning_rate = 2e-4
optimiser = torch.optim.SGD([beta, Z], lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = epochs // 100, gamma = 0.99)
loss_function_values = np.zeros(epochs)
for epoch in range(epochs):
    loss_function_values[epoch] = FitOneShot(dataset, Z, beta, optimiser, scheduler).item()
    print("Epoch:", epoch, "\t\tLearning rate:", "{:2e}".format(optimiser.param_groups[0]['lr']), "\t\tLoss:", round(loss_function_values[epoch],3))


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

pd.DataFrame(Z_long_format).to_csv(path+'output/positions.csv', index = False, header = False)
pd.DataFrame(loss_function_values).to_csv(path+'output/loss_function_values.csv', index = False, header = False)

#plt.plot(loss_function_values)
