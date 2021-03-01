#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:00:00 2020
@author: Marco and Riccardo
"""
###########
 ## FIT ##
###########

import math
import torch
from torch.utils.data import Dataset
from torch.distributions.normal import  Normal
RV = Normal(0,1)

### CLASSES AND METHODS

# CDataset: I am overwriting the methods __init__, __getitem__ and __len__,
class MDataset(Dataset):
    def __init__(self, timestamps, interactions, changepoints, transform=False, device = "cpu"):
        self.timestamps = timestamps
        self.interactions = interactions
        self.n_entries = len(timestamps)
        self.n_nodes = torch.max(interactions).item() + 1
        self.changepoints = changepoints
        self.n_changepoints = len(self.changepoints)
        self.segment_length = changepoints[1] - changepoints[0]
        self.transform = transform
        self.device = device
        
    def __getitem__(self, item):
        interaction = [self.timestamps[item], self.interactions[item,:]]
        if self.transform is not False:
            interaction = torch.Tensor(interaction)
            if self.device=="cuda":
                interaction = interaction.to(self.device)
        return interaction
    
    def __len__(self):
        return self.n_entries

def projection_model_negloglike(dataset, Z_in, penalty, device = "cpu"):
    # Make sure that we are dealing with nonnegative values only
    Z = Z_in**2
    
    # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
    prior = 0
    prior += penalty* torch.sum(Z[:,:,0]**2)
    prior += penalty * torch.sum((Z[:,:,1:(dataset.n_changepoints)] - Z[:,:,0:(dataset.n_changepoints-1)])**2)
    
    # This evaluates the poisson logrates at the timestamps when each of the interactions happen
    kappa = (dataset.timestamps // dataset.segment_length).long()
    deltas = (dataset.timestamps / dataset.segment_length - kappa).squeeze()
    one_minus_deltas = torch.add(-deltas, 1).squeeze()
    Z_sender_cur = Z[dataset.interactions[:,0:1],:,kappa].squeeze()
    Z_sender_new = Z[dataset.interactions[:,0:1],:,kappa+1].squeeze()
    Z_receiv_cur = Z[dataset.interactions[:,1:2],:,kappa].squeeze()
    Z_receiv_new = Z[dataset.interactions[:,1:2],:,kappa+1].squeeze()
    first_likelihood_term = torch.zeros(dataset.n_entries, dtype = torch.float64, device = device)
    first_likelihood_term += one_minus_deltas**2 * torch.sum(Z_sender_cur * Z_receiv_cur,1)
    first_likelihood_term += deltas * one_minus_deltas * torch.sum(Z_sender_cur * Z_receiv_new,1)
    first_likelihood_term += deltas * one_minus_deltas * torch.sum(Z_sender_new * Z_receiv_cur,1)
    first_likelihood_term += deltas**2 * torch.sum(Z_sender_new * Z_receiv_new,1)
    
    # This evaluates the value of the integral for the rate function, across all pairs of nodes and timeframes
    integral = 0
    for k in list(range(dataset.n_changepoints)[0:(dataset.n_changepoints-1)]):
        Z_cur = Z[:,:,k]
        Z_new = Z[:,:,k+1]
        Sij00 = ( torch.sum(torch.mm(Z_cur,Z_cur.t())) - torch.sum(Z_cur*Z_cur) ) / 6
        Sij11 = ( torch.sum(torch.mm(Z_new,Z_new.t())) - torch.sum(Z_new*Z_new) ) / 6
        Sij01 = ( torch.sum(torch.mm(Z_cur,Z_new.t())) - torch.sum(Z_cur*Z_new) ) / 12
        Sij10 = ( torch.sum(torch.mm(Z_new,Z_cur.t())) - torch.sum(Z_new*Z_cur) ) / 12
        integral += Sij00 + Sij11 + Sij01 + Sij10
    
    return prior - torch.sum(torch.log(first_likelihood_term)) + integral

def distance_model_negloglike(dataset, Z, beta, penalty, device = "cpu"):
    # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
    prior = 0
    prior += penalty * torch.sum(Z[:,:,0]**2)
    prior += penalty * torch.sum((Z[:,:,1:(dataset.n_changepoints)] - Z[:,:,0:(dataset.n_changepoints-1)])**2)
    
    # This evaluates the poisson logrates at the timestamps when each of the interactions happen
    kappa = (dataset.timestamps // dataset.segment_length).long()
    deltas = (dataset.timestamps / dataset.segment_length - kappa.double()).squeeze()
    one_minus_deltas = torch.add(-deltas, 1).squeeze()
    Z_sender_cur = Z[dataset.interactions[:,0:1],:,kappa].squeeze()
    Z_sender_new = Z[dataset.interactions[:,0:1],:,kappa+1].squeeze()
    Z_receiv_cur = Z[dataset.interactions[:,1:2],:,kappa].squeeze()
    Z_receiv_new = Z[dataset.interactions[:,1:2],:,kappa+1].squeeze()
    first_likelihood_term = torch.zeros(dataset.n_entries, dtype = torch.float64, device = device)
    first_likelihood_term += one_minus_deltas**2 * ( 2*torch.sum(Z_sender_cur * Z_receiv_cur,1) - torch.sum(Z_sender_cur * Z_sender_cur,1) - torch.sum(Z_receiv_cur * Z_receiv_cur,1) )
    first_likelihood_term += 2 * deltas * one_minus_deltas * (  torch.sum(Z_sender_cur * Z_receiv_new,1) + torch.sum(Z_sender_new * Z_receiv_cur,1) - torch.sum(Z_sender_cur * Z_sender_new,1) - torch.sum(Z_receiv_cur * Z_receiv_new,1)  )
    first_likelihood_term += deltas**2 * ( 2*torch.sum(Z_sender_new * Z_receiv_new,1) - torch.sum(Z_sender_new * Z_sender_new,1) - torch.sum(Z_receiv_new * Z_receiv_new,1) )
    
    # Integral of the rate function
    integral = 0.
    
    for k in list(range(dataset.n_changepoints)[0:(dataset.n_changepoints-1)]):
        tau_cur = dataset.changepoints[k]
        tau_new = dataset.changepoints[k+1]
        Z_cur = Z[:,:,k]
        Z_new = Z[:,:,k+1]
        tZ1  =  torch.sum(Z_cur**2, 1)
        tZ2 = tZ1.expand(tZ1.shape[0], tZ1.shape[0])
        tZ1 = tZ2.transpose(0,1)
        D = tZ1 + tZ2 -2*torch.mm(Z_cur, Z_cur.transpose(0,1))  # its element (i,j) is || z_i - z_j ||_2^2
        N = len(D)
        D_vec = D[torch.triu(torch.ones(N,N, device = device),1) == 1]
        DZ = Z_new - Z_cur            # This is \Delta 
        tDZ1 = torch.sum(DZ**2, 1)    
        tDZ2 = tDZ1.expand(tDZ1.shape[0], tDZ1.shape[0])
        tDZ1 = tDZ2.transpose(0,1)
        S = tDZ1 + tDZ2 -2*torch.mm(DZ, DZ.transpose(0,1))   # its element  (i,j) is || \Delta_i - \Delta_j ||_2^2
        S_vec = S[torch.triu(torch.ones(N,N, device = device),1) == 1]
        tA = torch.sum((DZ*Z_cur), 1)
        tB = tA.expand(tA.shape[0], tA.shape[0])
        tA = tB.transpose(0,1)
        C = torch.mm(DZ, Z_cur.transpose(0,1)) +  torch.mm(Z_cur, DZ.transpose(0,1)) - tA - tB
        C_vec = C[torch.triu(torch.ones(N,N, device = device),1) == 1]        
        mu = (1/S_vec)*(C_vec) 
        sigma = torch.sqrt(1/(2*S_vec))
        S = torch.exp(-(D_vec - S_vec*(mu**2)))*sigma*(RV.cdf((1 - mu)/sigma) - RV.cdf((0 - mu)/sigma))*(tau_new - tau_cur)
        integral += S.sum()        
    return prior - beta*len(dataset) - torch.sum(first_likelihood_term) + torch.sqrt(2*torch.tensor([math.pi], dtype=torch.float64, device = device))*torch.exp(beta)*integral

def FitOneShot(dataset, 
               Z, 
               optimiser, 
               penalty,
               beta = None, 
               scheduler=None, 
               device = "cpu", 
               model = "projection",
               ):
    optimiser.zero_grad()
    if model == 'distance':
        loss_function = distance_model_negloglike(dataset, Z, beta, penalty, device)
    elif model == 'projection':
        loss_function = projection_model_negloglike(dataset, Z, penalty, device)
    loss_function.backward()
    optimiser.step()
    if not scheduler == None:
        scheduler.step()
    return loss_function


############
 ## PLOT ##
############

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cividis = cm.get_cmap('cividis', 12)

from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os

def fade_node_colors(dataset, Z, bending = 1):
    """
    Defines the colors of the nodes as values between 0 and 1 at each changepoint -- this will be used in the animation
    @param      dataset         the observed data
    @param      Z               the estimated latent positions at each of the changepoints
    @param      bending         the values are amplified or reduced according to this positive value: >1 will increase variability, <1 will decrease variability
    """
    colors = np.zeros((dataset.n_nodes, dataset.n_changepoints))
    for t in range(dataset.n_changepoints):
        cZ = Z[:,:,t]
        print('changepoint: ', t)
        tDZ1 = torch.sum(cZ**2, 1)    
        tDZ2 = tDZ1.expand(tDZ1.shape[0], tDZ1.shape[0])
        tDZ1 = tDZ2.transpose(0,1)
        S = (1.0 + tDZ1 + tDZ2 - 2*torch.mm(cZ, cZ.transpose(0,1)))
        S[range(len(S)), range(len(S))] = torch.zeros(len(S), dtype = torch.float64)
        colors[:,t] = torch.mean(S,1)
        #for i in range(dataset.n_nodes):
        #    for j in range(dataset.n_nodes):
        #        if i != j:
        #            colors[i,t] += (  1 + (Z[i,0,t]-Z[j,0,t])**2 + (Z[i,1,t]-Z[j,1,t])**2  ) / (dataset.n_nodes-1)
    colors = colors**bending
    colors /= colors.min()
    colors -= 1
    colors /= colors.max()# now we have values between 0 and 1
    colors *= -1
    colors += 1# now we have 1 minus the previous values
    return colors

def fade_node_sizes(dataset, bending = 1):
    """
    Defines the sizes of the nodes as values between 0 and 1 at each changepoint -- this will be used in the animation
    @param      dataset         the observed data
    @param      bending         the values are amplified or reduced according to this positive value: >1 will increase variability, <1 will decrease variability
    """
    sizes = np.ones((dataset.n_nodes, dataset.n_changepoints))
    for edge in range(dataset.n_entries):
        i = dataset.interactions[edge,0].item()
        j = dataset.interactions[edge,1].item()
        kappa = (dataset.timestamps[edge] // dataset.segment_length).int().item()
        sizes[i,kappa] += 2
        sizes[j,kappa] += 2
        if kappa >= 1: 
            sizes[i,kappa-1] += 1
            sizes[j,kappa-1] += 1
        if kappa <= dataset.n_changepoints - 2: 
            sizes[i,kappa+1] += 1
            sizes[j,kappa+1] += 1
    sizes = sizes**bending
    sizes /= sizes.min()
    sizes -= 1
    sizes /= sizes.max()
    return sizes

def create_snaps(Z, changepoints, frames_btw, node_colors, node_sizes, model_type, dpi = 100, times = None, node_to_track = None):
    """
    Creates a sequence of images that will compose the video.
    @param      Z               the latent positions from the output of the fitting algorithm
    @param      changepoints    the changepoints that are being used
    @param      frames_btw      the number of additional frames that are inserted inbetween any two changepoints, to make the transitions smoother
    @param      node_colors     how the nodes should be colored at each changepoint
    @param      node_sizes      the size of the nodes at each changepoint
    @param      dpi             resolution of the exported pdf images
    @param      model_type      a string, either 'projection' or 'distance' 
    @param      times           optional time labels (dates) to plot in the title of each snap
    @return                     a list of strings indicating the path to the images that must be collated in the video
    """
    n_nodes = Z.shape[0]
    n_dim = Z.shape[1]
    n_cps = Z.shape[2]
    n_frames = frames_btw * (n_cps-1) + n_cps
    cps_large = np.zeros(n_frames)
    cps_large[n_frames-1] = changepoints[n_cps-1]
    pos = np.zeros((n_nodes, n_dim, n_frames))
    pos[:,:,n_frames-1] = Z[:,:,n_cps-1]
    colors_large = np.zeros((n_nodes, n_frames))
    colors_large[:,n_frames-1] = node_colors[:,n_cps-1]
    sizes_large = np.zeros((n_nodes, n_frames))
    sizes_large[:,n_frames-1] = node_sizes[:,n_cps-1]
    for frame in range(n_frames-1):
        print('(1) - frame: ', frame)
        cp0 = frame // (frames_btw+1)
        cp1 = (frame // (frames_btw+1)) + 1
        delta = (frame % (frames_btw+1)) / (frames_btw+1)
        cps_large[frame] = (1-delta)*changepoints[cp0] + delta*changepoints[cp1]
        for i in range(n_nodes):
            pos[i,0,frame] = (1-delta)*Z[i,0,cp0] + delta*Z[i,0,cp1]
            pos[i,1,frame] = (1-delta)*Z[i,1,cp0] + delta*Z[i,1,cp1]
            colors_large[i,frame] = (1-delta)*node_colors[i,cp0] + delta*node_colors[i,cp1]
            sizes_large[i,frame] = (1-delta)*node_sizes[i,cp0] + delta*node_sizes[i,cp1]
    pos_limit = np.abs(pos).max()
    for frame in range(n_frames-1):
        print('(2) - frame: ', frame)
        plt.figure()
        if times == None:
            plt.title("Latent Positions at time " + str(round(cps_large[frame],2)), loc = "left")
        else:
            plt.title("Latent Positions at time " + times[frame], loc = "left")
        plt.xlim((-pos_limit,pos_limit))
        plt.ylim((-pos_limit,pos_limit))
        for idi in range(n_nodes):        
            if idi != node_to_track:
                if frame >= 2: plt.plot([pos[idi,0,frame-2], pos[idi,0,frame-1]], [pos[idi,1,frame-2], pos[idi,1,frame-1]], 'k-', alpha = 0.2, color = cividis(colors_large[idi,frame]))
                if frame >= 1: plt.plot([pos[idi,0,frame-1], pos[idi,0,frame-0]], [pos[idi,1,frame-1], pos[idi,1,frame-0]], 'k-', alpha = 0.3, color = cividis(colors_large[idi,frame]))
                #if frame < n_frames-1: plt.plot([pos[idi,0,frame+0], pos[idi,0,frame+1]], [pos[idi,1,frame+0], pos[idi,1,frame+1]], 'k-', alpha = 0.3)
                #if frame < n_frames-2: plt.plot([pos[idi,0,frame+1], pos[idi,0,frame+2]], [pos[idi,1,frame+1], pos[idi,1,frame+2]], 'k-', alpha = 0.1)
                plt.plot(pos[idi,0,frame], pos[idi,1,frame], 'bo', color = 'blue', markersize = 1 + sizes_large[idi,frame] * 8, markeredgewidth = 0.2, alpha = 0.4, markerfacecolor =cividis(colors_large[idi,frame]))
            else:
                if frame >= 2: plt.plot([pos[idi,0,frame-2], pos[idi,0,frame-1]], [pos[idi,1,frame-2], pos[idi,1,frame-1]], 'k-', alpha = 0.4, color = 'red')
                if frame >= 1: plt.plot([pos[idi,0,frame-1], pos[idi,0,frame-0]], [pos[idi,1,frame-1], pos[idi,1,frame-0]], 'k-', alpha = 0.8, color = 'red')
                plt.plot(pos[idi,0,frame], pos[idi,1,frame], 'bo', color = 'blue', markersize = 1 + sizes_large[idi,frame] * 8, markeredgewidth = 0.2, alpha = 1, markerfacecolor= 'red')
                
        plt.savefig('results_'+model_type+'/snaps/snap_'+str(frame)+'.png', dpi = dpi)
        plt.close()
    images = []
    for i in range(n_frames-1):
        images.append('results_'+model_type+'/snaps/snap_'+str(i)+'.png')
    return images

def make_video(outvid, images, outimg = None, fps = 2, size = (600,450), is_color = True, format = "mp4v"):
    """
    Create a video from a list of images.
    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid

def clpm_animation(outvid, Z, changepoints, frames_btw, node_colors, node_sizes, dpi, period, size = (1200,900), is_color = True, format = "mp4v", times = None, node_to_track = None, model_type = "distance"):
    """
    Combines the functions create_snaps and make_video to produce the animation for a fitted clpm
    @param      outvid          output video
    @param      Z               the latent positions from the output of the fitting algorithm
    @param      changepoints    the changepoints that are being used
    @param      frames_btw      the number of additional frames that are inserted inbetween any two changepoints, to make the transitions smoother
    @param      node_colors     how the nodes should be colored at each changepoint
    @param      node_sizes      the size of the nodes at each changepoint
    @param      dpi             resolution of the exported pdf images
    @param      period          time it takes on video to progress from one changepoint to the next
    @param      size            see make_video
    @param      is_color        see make_video
    @param      format          see make_video
    @param      times           optional time labels (dates) to plot in the title of each snap
    @param      model_type      a string. Either "distance" (default) or "projection"
    @return                     see make_video
    """
    images = create_snaps(Z, changepoints, frames_btw, node_colors, node_sizes, model_type, dpi, times, node_to_track)
    fps = (frames_btw+1) / period
    return make_video(outvid, images, None, fps, size, is_color, format)


##################################################
  ## Main functions to be called by the user ###
##################################################


def ClpmFit(epochs, 
            n_changepoints, 
            model_type = 'distance',
            penalty = 10.,
            lr_Z = 1e-3, 
            lr_beta = 1e-7, 
            device = 'cpu'):  
    """

    Parameters
    ----------
    epochs : Number of epochs.
    n_changepoints : Number of changepoints in the latent trajectories.
    model_type : Either 'distance' or 'projection' The default is 'distance'.
    lr_Z : Default (fixed) learning rate for Z. The default is 1e-3.
    lr_beta :Defalut (fixed) learning rate for beta, ingored if model_type is 'projection'. The default is 1e-7.
    device : 'cpu' or 'cuda'. The default is 'cpu'.

    Returns
    -------
    None. This is a void function printing in the folder "output/model_type" four files:
        1) changepoints.csv
        2) loss.pdf
        3) loss_function_values.csv
        4) positions.csv (The estimated Z)

    """
    
    import numpy as np
    import pandas as pd
    import torch
    import os
    import sys
    
    if (model_type != 'distance' and model_type!= 'projection'):
        print("fatal error: not supported model_type", file=sys.stderr)

        
    path = ""
    my_step = 1/n_changepoints
    learning_rate = lr_Z  # 1e-2
    
    ####################################################
    ## looks for existing directories or creates them ##
    ####################################################
    
    list_files = os.listdir()
    
    if not 'output_projection' in list_files:
        os.mkdir('output_projection')    
    if not 'output_distance' in list_files:
        os.mkdir('output_distance')
    if not 'results_projection' in list_files:
        os.mkdir('results_projection')    
        os.mkdir('results_projection/snaps')
    if not 'results_distance' in list_files:
        os.mkdir('results_distance')
        os.mkdir('results_distance/snaps')    
    
    
    #### DATA AND PARAMETER INITIALISATION
    edgelist = pd.read_csv(path + 'edgelist.csv')
    time_max = np.max(edgelist.iloc[:,0])
    changepoints = np.arange(start = 0.0, stop = 1.0 + my_step ,  step = my_step)*(time_max + 0.0001)
    # L'ultimo change point deve essere più alto del più alto interaction time
    changepoints = torch.tensor(changepoints, dtype = torch.float64, device = device) 
    changepoints.reshape(-1,1)
    np.savetxt(path + "output_" + model_type + "/changepoints.csv", changepoints.cpu(), delimiter = ',')
    n_changepoints = len(changepoints)
    timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
    interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long, device = device)
    n_nodes = torch.max(interactions).item() + 1
    dataset = MDataset(timestamps, interactions, changepoints, transform = True, device = device)
    Z = torch.tensor(np.random.normal(size = (n_nodes,2,(n_changepoints))), dtype = torch.float64, device = device, requires_grad = True)
    beta = torch.tensor(np.random.normal(size = 1), dtype = torch.float64, device = device, requires_grad = True) 
    
    ### OPTIMISATION
    optimiser = torch.optim.SGD([{'params': beta, "lr": lr_beta},
                                  {'params': Z},], 
                                lr=learning_rate)
    #optimiser = torch.optim.SGD([beta,Z], lr = learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = epochs // 10, gamma = 0.995)
    loss_function_values = np.zeros(epochs)
    for epoch in range(epochs):
        loss_function_values[epoch] = FitOneShot(dataset, Z, beta = beta, optimiser=optimiser, device = device, model = model_type, penalty = penalty).item()
        if model_type == "distance":
            print("Epoch:", epoch, "\t\tLR (beta):", "{:2e}".format(optimiser.param_groups[0]['lr']), "\t\tLR (Z):", "{:2e}".format(optimiser.param_groups[1]['lr']), "\t\tLoss:", round(loss_function_values[epoch],3))
        else:
            print("Epoch:", epoch, "\t\tLR (Z):", "{:2e}".format(optimiser.param_groups[1]['lr']), "\t\tLoss:", round(loss_function_values[epoch],3))
            
        
    ### Plotting the loss
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_function_values)
    plt.savefig(path+'output_'+ model_type +'/loss.pdf')
    plt.close()    
    
    
    ### Exporting the OUTPUT Z
    
    Z_long_format = np.zeros([Z.shape[0]*Z.shape[1]*Z.shape[2], 4])
    index = 0
    for i in range(Z.shape[0]):
    	for d in range(Z.shape[1]):
    		for t in range(Z.shape[2]):
    			Z_long_format[index, 0] = i+1
    			Z_long_format[index, 1] = d+1
    			Z_long_format[index, 2] = t+1
    			Z_long_format[index, 3] = Z[i,d,t]
    			index += 1
    
    pd.DataFrame(Z_long_format).to_csv(path+'output_'+model_type+'/positions.csv', index = False, header = False)
    pd.DataFrame(loss_function_values).to_csv(path+'output_'+model_type+'/loss_function_values.csv', index = False, header = False)
    

###############################################################################
    
    
def ClpmPlot(model_type = 'distance',
             dpi = 250,
             period = 1,
             size = (1200,900),
             is_color = True,
             formato = 'mp4v',
             frames_btw = 5 
        ):
    '''
    

    Parameters
    ----------
    model_type : Either 'distance' or 'projection'. The default is 'projection'.
    dpi : The resolution of the exported image. The default is 250.
    period : Time it takes on video to progress from one changepoint to the next. The default is 1.
    size : Image resolution. The default is (1200,900).
    is_color : Boolean, the default is True.
    formato : See make video. The default is 'mp4v'.
    frames_btw : How many interpolations to produce between to consecutive changepoints. The default is 5.

    Returns
    -------
    The dynamic graph embedding video in the results/model_type/ folder.

    '''
    
    import pandas as pd
    import numpy as np
    import torch
    import os, sys
    
    ######################################################
    ## looks for existing directories and prepare them ##
    ######################################################
    
    list_files = os.listdir()
    
    if model_type == 'projection':
        if not 'output_projection' in list_files:
            print('fatal error: no output directory detected!', sys.stderr)
        if 'results_projection' in list_files:
            sub_list_files = os.listdir('results_projection/')
            if 'snaps' in sub_list_files:
                if len(os.listdir('results_projection/snaps'))>0:
                    for item in os.listdir('results_projection/snaps'):
                        if item.endswith('.png'):
                            os.remove(os.path.join('results_projection/snaps/', item))
            else:
                os.mkdir('results_projection/snaps')        
        if not 'results_projection' in list_files:
            os.mkdir('results_projection')    
            os.mkdir('results_projection/snaps')
        
    if model_type == 'distance'  :  
        if not 'output_distance' in list_files:
            print('fatal error: no output directory detected!', sys.stderr)    
        if 'results_distance' in list_files:
            sub_list_files = os.listdir('results_distance/')
            if 'snaps' in sub_list_files:
                if len(os.listdir('results_distance/snaps'))>0:
                    for item in os.listdir('results_distance/snaps'):
                        if item.endswith('.png'):
                            os.remove(os.path.join('results_distance/snaps/', item))                
            else:
                os.mkdir('results_distance/snaps')         
        if not 'results_distance' in list_files:
            os.mkdir('results_distance')
            os.mkdir('results_distance/snaps')  
    
    
    
    folder = ''
    
    edgelist = pd.read_csv('edgelist.csv')
    changepoints = np.loadtxt(folder+"output_"+model_type+"/changepoints.csv", delimiter = ',')
    n_changepoints = len(changepoints)
    timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64)
    interactions = torch.tensor(edgelist.iloc[:,1:3].values, dtype = torch.long)
    n_nodes = torch.max(interactions).item() + 1
    dataset = MDataset(timestamps, interactions, changepoints, transform = True)
    Z = torch.zeros(size = (n_nodes,2,(n_changepoints)), dtype = torch.float64)
    Z_long = pd.read_csv(folder+'output_'+model_type+'/positions.csv', header = None)
    for row in range(len(Z_long)):
        i = Z_long.iloc[row,0]-1
        d = Z_long.iloc[row,1]-1
        t = Z_long.iloc[row,2]-1
        val = Z_long.iloc[row,3]
        Z[i.astype('int'),d.astype('int'),t.astype('int')] = val
    
    if model_type == 'projection':
        Z = Z**2
    
    # times
    from datetime import datetime
    
    outvid = folder + 'results_'+model_type+'/video.mp4'
    node_colors = fade_node_colors(dataset, Z, bending = 1)
    node_sizes = fade_node_sizes(dataset, bending = 1)
    
    
    clpm_animation(outvid, 
                   Z.detach().numpy(), 
                   changepoints, 
                   frames_btw, 
                   node_colors, 
                   node_sizes,
                   dpi, period, size, is_color, formato, node_to_track = 0, model_type=model_type)







