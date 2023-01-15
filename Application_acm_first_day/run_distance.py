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

plots_only = True
verbose = True

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

edge_list = pd.read_csv('edgelist.csv')
network = NetworkCLPM(edge_list, verbose, device='cpu')

n_change_points = 20
model_type = 'distance'
penalty = 20.
model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)

n_epochs = 8000
batch_size = 100
lr_z = 1e-5
lr_beta = 1e-7

if plots_only is False:
    model.fit(network, n_epochs, batch_size, lr_z, lr_beta)
    model.export_fit()
else: model.import_fit()

frames_btw = 50
cluster_n_groups = 5
plot_opt = {"period": 1,
            "frames_btw": frames_btw,
            "coloring_method": "cluster",
            "coloring_n_groups": cluster_n_groups,
            "time_format": "%H:%M:%S",
            "start_date": (2009, 6, 29, 8, 0),
            "end_date": (2009, 6, 29, 20, 59)}
model.def_plot_pars(plot_opt)
model.create_animation(True)

model.clusteredness_index([0.1, 0.2, 0.3], 8, 21, frames_btw)
