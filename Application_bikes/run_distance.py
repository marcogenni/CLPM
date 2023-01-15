#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('../')

from CLPM_dataset import *
from CLPM import *

np.random.seed(12345)
torch.manual_seed(54321)

plots_only = False
verbose = True

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

edge_list = pd.read_csv('edgelist.csv')
network = NetworkCLPM(edge_list, verbose, device='cpu')

n_change_points = 96
model_type = 'distance'
penalty = 10.
model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)

n_epochs = 300
batch_size = 78
lr_z = 1e-4
lr_beta = 1e-7

if plots_only is False:
    model.fit(network, n_epochs, batch_size, lr_z, lr_beta)
    model.export_fit()
else: model.import_fit()

model.reduce_network(edgelist=edge_list, n_hubs=2, type_of='degree', n_sub_nodes=60)

frames_btw = 10
cluster_n_groups = 5
plot_opt = {"period": 0.5,
            "frames_btw": frames_btw,
            "nodes_to_track": [0, 22, 50],
            "time_format": '%H:%M:%S',
            "start_date": [2015, 9, 6, 0, 0],
            "end_date": [2015, 9, 6, 23, 59]}
model.def_plot_pars(plot_opt)
model.create_animation(True)

model.clusteredness_index([0.05, 0.2], 0, 24, frames_btw)


