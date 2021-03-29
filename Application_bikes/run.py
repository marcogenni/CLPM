#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from CLPM_functions import *

np.random.seed(12345)
torch.manual_seed(54321)

epochs = 500
n_changepoints = 96
period = 1
frames_btw = 20

snapshot_times = ['2015/09/06 04:03:00',
                  '2015/09/06 07:57:32',
                  '2015/09/06 17:03:37',
                  '2015/09/06 19:46:05'
                  ]
 
fnames = ['app_bikes_dist_'+ str(idx) +'.pdf' for idx in range(len(snapshot_times))]                  


ClpmFit(epochs = epochs, 
        n_changepoints = n_changepoints, 
        model_type = 'distance',
        penalty = 20.,
        lr_Z = 1e-4, 
        lr_beta = 2e-6, 
        device = 'cuda')

ClpmPlot(model_type = 'distance',
         dpi = 250,
         period = period,
         size = (1200,900),
         is_color = True,
         formato = 'mp4v',
         frames_btw = frames_btw,
         nodes_to_track = [0,22,50],
         sub_graph = True,
         type_of = 'degree',
         n_hubs = 2,
         n_sub_nodes = 60,
         start_date = [2015, 9, 6, 0, 0],
         end_date = [2015, 9, 6, 23, 59],
         time_format = '%Y/%m/%d %H:%M:%S')

ClpmSnap(model_type = 'distance',
         extraction_times=snapshot_times,
         filenames=fnames,
         dpi = 250,
         period = period,
         size = (1200,900),
         is_color = True,
         formato = 'mp4v',
         frames_btw = frames_btw,
         nodes_to_track = [0,22,50],
         sub_graph = True,
         type_of = 'degree',
         n_hubs = 2,
         n_sub_nodes = 60,
         start_date = [2015, 9, 6, 0, 0],
         end_date = [2015, 9, 6, 23, 59],
         time_format = '%Y/%m/%d %H:%M:%S')

