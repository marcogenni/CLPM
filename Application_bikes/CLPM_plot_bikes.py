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
    i = Z_long.iloc[row,0]-1
    d = Z_long.iloc[row,1]-1
    t = Z_long.iloc[row,2]-1
    val = Z_long.iloc[row,3]
    Z[i.astype('int'),d.astype('int'),t.astype('int')] = val

# times
from datetime import datetime


##    

print('I begin plotting...')

outvid = folder + 'results/video.mp4'
frames_btw = 4 #20
print('Fade node colors ... ')
node_colors = fade_node_colors(dataset, Z, bending = 1)
node_sizes = fade_node_sizes(dataset, bending = 1)
dpi = 250
period = 1
size = (1200,900)
is_color = True
formato = 'mp4v'

n_cps = len(changepoints)
n_frames = (n_cps-1)*frames_btw + n_cps
now = datetime(2000, 1, 1, 0, 0)
last = datetime(2000, 1, 2, 0, 0)
delta = (last - now)/(n_frames-1)
times = []
while now < last:
    times.append(now.strftime('%H:%M:%S'))
    now += delta


clpm_animation(outvid, Z.cpu().detach().numpy(),
               changepoints, 
               frames_btw, 
               node_colors, 
               node_sizes, 
               dpi,
               period,
               size, is_color, formato, times)


