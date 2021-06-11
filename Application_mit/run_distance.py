#!/usr/bin/env python3


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

n_change_points = 20
model_type = 'distance'
penalty = 10.
model = ModelCLPM(network, n_change_points, model_type, penalty, verbose)

n_epochs = 5000
batch_size = 60
lr_z = 1e-4
lr_beta = 1e-7
model.fit(network, n_epochs, batch_size, lr_z, lr_beta)

model.export()

period = 2.5
frames_btw = 80
ClpmPlot(model_type=model_type,
         dpi=250,
         period=period,
         size=(1200, 900),
         is_color=True,
         formato='mp4v',
         frames_btw=frames_btw,
         nodes_to_track=[None],
         sub_graph=False,
         type_of='friendship',
         n_hubs=2,
         n_sub_nodes=100,
         start_date=[2004, 9, 14, 0, 0],
         end_date=[2005, 5, 5, 0, 0],
         time_format='%Y/%m/%d')

