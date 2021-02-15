import pandas as pd
import numpy as np
import torch

device = 'cpu'

import sys
sys.path.append('../')
from CLPM_fit import *
from CLPM_plot import *
folder = ''

#############################
 ## Auxiliary function(s) ##
############################# 
def get_sub_graph(edgelist, n_hubs=2, type_of = 'friendship', n_sub_nodes = 100):
    n_nodes = np.max(edgelist['receiver']) + 1
    Adj = np.zeros(shape = (n_nodes, n_nodes))
    for idx in range(len(edgelist)):
        sender = edgelist.iloc[idx,1]
        receiver = edgelist.iloc[idx,2]
        Adj[sender, receiver] += 1
        Adj[receiver, sender] += 1
    
    deg = np.sum(Adj, axis = 1)    
    if type_of == 'friendship':
        # Three most active nodes
        hubs = np.argsort(-deg)[0:n_hubs]  
        sAdj1 = Adj[hubs,:]  
        tmp = np.sum(sAdj1, 0) 
        sub_nodes = np.where(tmp != 0)
        pos_1 = np.isin(edgelist['sender'], sub_nodes)
        pos_2 = np.isin(edgelist['receiver'], sub_nodes)
        pos_3 = pos_1 & pos_2 
        pos_final = np.where(pos_3 == True)[0]
        edgelist = edgelist.iloc[pos_final, :]
        return sub_nodes[0], edgelist
    elif type_of == 'degree':
        sub_nodes = np.argsort(-deg)[0:n_sub_nodes]
        sub_nodes = np.sort(sub_nodes)
        pos_1 = np.isin(edgelist['sender'], sub_nodes)
        pos_2 = np.isin(edgelist['receiver'], sub_nodes)
        pos_3 = pos_1 & pos_2 
        pos_final = np.where(pos_3 == True)[0]
        edgelist = edgelist.iloc[pos_final, :]
        return sub_nodes, edgelist
        

def edgelist_conversion(edgelist_, sub_nodes, n_nodes):
    """
    This function takes the new edgelist output by get_sub_graph and renames both senders and receivers in  such a way to have a continuous list of nodes
    
    Parameters
    ----------
    edgelist_ : The new edgelist related to the subgraph extracted by get_sub_graph.
    sub_nodes : The subset of the original nodes' set forming the sub-graph.
    n_nodes : The number of nodes in the **orginal** graph.

    Returns
    -------
    The new edgelist with renamed senders/receivers and a conversion table for nodes.

    """
    new_n_nodes = len(sub_nodes)
    conversion = np.repeat(-1,n_nodes)
    conversion[sub_nodes] = np.arange(0,new_n_nodes)
    edgelist_['sender'] = conversion[edgelist_['sender'].values]
    edgelist_['receiver'] = conversion[edgelist_['receiver'].values]
    return edgelist_, conversion

## most active nodes: 13, 153, 373    
    
edgelist = pd.read_csv('input/edgelist.csv')
n_nodes = np.max(edgelist.iloc[:,1:3].values)+1

sub_nodes, edgelist_ = get_sub_graph(edgelist.copy(), type_of = 'degree')
edgelist_, conversion = edgelist_conversion(edgelist_,sub_nodes,780) 

n_changepoints = 96
time_max = 24 # in this application we know the true value
changepoints = np.loadtxt("output/changepoints.csv", delimiter = ',')
timestamps = torch.tensor(edgelist_.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist_.iloc[:,1:3].values, dtype = torch.long, device = device)
new_n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)

Z = torch.zeros(size = (n_nodes,2,(n_changepoints)), dtype = torch.float64)
Z_long = pd.read_csv(folder+'output/positions.csv', header = None)
for row in range(len(Z_long)):
    i = Z_long.iloc[row,0]-1
    d = Z_long.iloc[row,1]-1
    t = Z_long.iloc[row,2]-1
    val = Z_long.iloc[row,3]
    Z[i.astype('int'),d.astype('int'),t.astype('int')] = val
    
Z = Z[sub_nodes,:,:]    

# times
from datetime import datetime


##    

print('I begin plotting...')

outvid = folder + 'results/video.mp4'
frames_btw = 10 #20
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
               np.exp(node_sizes)-.5, 
               dpi,
               period,
               size, is_color, formato, times, node_to_track=77)


