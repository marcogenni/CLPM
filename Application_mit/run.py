#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from CLPM_functions import *

np.random.seed(12345)
torch.manual_seed(54321)

epochs = 5000
n_changepoints = 20
period = 2.5
frames_btw = 80

snapshot_times = ['2004/10/02',
                  '2004/11/22', 
                  '2004/12/25', 
                  '2005/01/26', 
                  '2005/02/23', 
                  '2005/04/10']


ClpmFit(epochs = epochs, 
        n_changepoints = n_changepoints, 
        model_type = 'projection',
        penalty = 10.,
        lr_Z = 1e-4, 
        lr_beta = 1e-7, 
        device = 'cpu')

ClpmPlot(model_type = 'projection',
         dpi = 250,
         period = period,
         size = (1200,900),
         is_color = True,
         formato = 'mp4v',
         frames_btw = frames_btw,
         nodes_to_track = [None],
         sub_graph = False,
         type_of = 'friendship',
         n_hubs = 2,
         n_sub_nodes = 100,
         start_date = [2004, 9, 14, 0, 0],
         end_date = [2005, 5, 5, 0, 0],
         time_format = '%Y/%m/%d')

ClpmFit(epochs = epochs, 
        n_changepoints = n_changepoints, 
        model_type = 'distance',
        penalty = 10.,
        lr_Z = 1e-4, 
        lr_beta = 1e-7, 
        device = 'cpu')

ClpmPlot(model_type = 'distance',
         dpi = 250,
         period = period,
         size = (1200,900),
         is_color = True,
         formato = 'mp4v',
         frames_btw = frames_btw,
         nodes_to_track = [None],
         sub_graph = False,
         type_of = 'friendship',
         n_hubs = 2,
         n_sub_nodes = 100,
         start_date = [2004, 9, 14, 0, 0],
         end_date = [2005, 5, 5, 0, 0],
         time_format = '%Y/%m/%d')

ClpmSnap(model_type = 'distance',
         extraction_times=snapshot_times,
         dpi = 250,
         period = period,
         size = (1200,900),
         is_color = True,
         formato = 'mp4v',
         frames_btw = frames_btw,
         nodes_to_track = [None],
         sub_graph = False,
         type_of = 'friendship',
         n_hubs = 2,
         n_sub_nodes = 100,
         start_date = [2004, 9, 14, 0, 0],
         end_date = [2005, 5, 5, 0, 0],
         time_format = '%Y/%m/%d')

