#!/usr/bin/env python3


import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('../')

from CLPM_dataset import *
from CLPM_model import *
from CLPM_plot import *

np.random.seed(12345)
torch.manual_seed(54321)

verbose = False

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
  
tot_n_nodes = [30] #, 60, 90, 120, 150, 180]

store_times = np.zeros(shape=(2,len(tot_n_nodes)))
store_epochs = np.zeros(len(tot_n_nodes))

counter = 0

for n_nodes in tot_n_nodes:
  
  print('Working with {} nodes'.format(n_nodes))
  edge_list = pd.read_csv('edgelist_{}.csv'.format(n_nodes))
  network = NetworkCLPM(edge_list, verbose, device = device)
  
  n_change_points = 10
  model_type = 'distance'
  penalty = 20.
  
  model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)
  n_epochs = 100
  batch_size = n_nodes
  lr_z = 1e-4
  lr_beta = 1e-7
  model.fit(network, n_epochs, batch_size, lr_z, lr_beta)
  
  model.export()
  
  elapsed_secs = np.genfromtxt('output_distance/elapsed_secs.csv')
  store_times[0, counter] = elapsed_secs
    
  l_value = np.genfromtxt('output_distance/loss_function_values.csv')
  threshold = l_value[-1]
  
  model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)
  n_epochs = 1000
  batch_size = 3
  lr_z = 1e-4
  lr_beta = 1e-7
  model.fit(network, n_epochs, batch_size, lr_z, lr_beta, threshold)
  
  model.export()
 
  elapsed_secs = np.genfromtxt('output_distance/elapsed_secs.csv')
  store_times[1,counter] = elapsed_secs
  
  actual_epochs = np.genfromtxt('output_distance/n_epochs.csv')
  store_epochs[counter] = actual_epochs
  
  counter += 1
   
 



