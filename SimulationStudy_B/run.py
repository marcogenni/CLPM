#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from CLPM_functions import *

np.random.seed(12345)
torch.manual_seed(54321)

epochs = 5000
period = 2
frames_btw = 60

ClpmFit(epochs = epochs, 
        n_changepoints = 10, 
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
         start_date = None,
         end_date = None)

ClpmFit(epochs = epochs, 
        n_changepoints = 10, 
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
         start_date = None,
         end_date = None)

tte = [1.0, 4.3456, 7.5373]
ClpmSnap(extraction_times=tte,
             model_type = 'distance',
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
             start_date = None,
             end_date = None)



