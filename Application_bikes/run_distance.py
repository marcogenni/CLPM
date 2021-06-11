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

verbose = True

edge_list = pd.read_csv('edgelist.csv')
network = NetworkCLPM(edge_list, verbose)

n_change_points = 96
model_type = 'distance'
penalty = 30.
model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)

n_epochs = 500
batch_size = 780
lr_z = 1e-4
lr_beta = 1e-6
model.fit(network, n_epochs, batch_size, lr_z, lr_beta)

model.export()

period = 1
frames_btw = 40
ClpmPlot(model_type=model_type,
         dpi=250,
         period=period,
         size=(1200, 900),
         is_color=True,
         formato='mp4v',
         frames_btw=frames_btw,
         nodes_to_track=[0, 22, 50],
         sub_graph=True,
         type_of='degree',
         n_hubs=2,
         n_sub_nodes=60,
         start_date=[2015, 9, 6, 0, 0],
         end_date=[2015, 9, 6, 23, 59],
         time_format='%Y/%m/%d %H:%M:%S')


