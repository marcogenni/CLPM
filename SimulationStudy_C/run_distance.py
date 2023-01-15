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

edge_list = pd.read_csv('edgelist.csv', header=None)
network = NetworkCLPM(edge_list, verbose, device='cpu')

n_change_points = 10
model_type = 'distance'
penalty = 300.
model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)

n_epochs = 500
batch_size = 10
lr_z = 1e-5
lr_beta = 1e-6

if plots_only is False:
    model.fit(network, n_epochs, batch_size, lr_z, lr_beta)
    model.export_fit()
else: model.import_fit()

plot_opt = {"period": 1,
            "frames_btw": 20}
model.def_plot_pars(plot_opt)
model.create_animation(True)
