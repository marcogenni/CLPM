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

n_changepoints = 96
time_max = 24 # in this application we know the true value
changepoints = np.loadtxt("output/changepoints.csv", delimiter = ',')
timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long, device = device)
n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)

Z = torch.zeros(size = (n_nodes,2,(n_changepoints)), dtype = torch.float64)
Z_long = pd.read_csv(folder+'output/positions.csv', header = None)
for row in range(len(Z_long)):
    i = Z_long[row][0]-1
    d = Z_long[row][1]-1
    t = Z_long[row][2]-1
    val = Z_long[row][3]
    Z[i,d,t] = val
