#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:00:00 2020
@author: Marco and Riccardo
"""

import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader  

device = "cpu"

from torch.distributions.normal import  Normal
RV = Normal(0,1)

### CLASSES AND METHODS

# CDataset: I am overwriting the methods __init__, __getitem__ and __len__,
class MDataset(Dataset):
    def __init__(self, timestamps, interactions, changepoints, transform=False):
        self.timestamps = timestamps
        self.interactions = interactions
        self.n_entries = len(timestamps)
        self.n_nodes = torch.max(interactions).item() + 1
        self.changepoints = changepoints
        self.n_changepoints = len(self.changepoints)
        self.segment_length = changepoints[1] - changepoints[0]
        self.transform = transform
        
    def __getitem__(self, item):
        interaction = [self.timestamps[item], self.interactions[item,:]]
        if self.transform is not False:
            interaction = torch.Tensor(interaction)
            if device=="cuda":
                interaction = interaction.to(device)
        return interaction
    
    def __len__(self):
        return self.n_entries

def projection_model_negloglike(dataset, Z):
    # Make sure that we are dealing with nonnegative values only
    Z = Z.abs()
    
    # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
    prior = 0
    prior += 5 * torch.sum(Z[:,:,0]**2)
    prior += 5 * torch.sum((Z[:,:,1:(dataset.n_changepoints)] - Z[:,:,0:(dataset.n_changepoints-1)])**2)
    
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
    for k in list(range(dataset.n_changepoints)[0:(n_changepoints-1)]):
        Z_cur = Z[:,:,k]
        Z_new = Z[:,:,k+1]
        Sij00 = ( torch.sum(torch.mm(Z_cur,Z_cur.t())) - torch.sum(Z_cur*Z_cur) ) / 6
        Sij11 = ( torch.sum(torch.mm(Z_new,Z_new.t())) - torch.sum(Z_new*Z_new) ) / 6
        Sij01 = ( torch.sum(torch.mm(Z_cur,Z_new.t())) - torch.sum(Z_cur*Z_new) ) / 12
        Sij10 = ( torch.sum(torch.mm(Z_new,Z_cur.t())) - torch.sum(Z_new*Z_cur) ) / 12
        integral += Sij00 + Sij11 + Sij01 + Sij10
    
    return prior - torch.sum(torch.log(first_likelihood_term)) + integral

def clpm_negloglike(dataset, Z):
    # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
    prior = 0
    prior += 5 * torch.sum(Z[:,:,0]**2)
    prior += 5 * torch.sum((Z[:,:,1:(dataset.n_changepoints)] - Z[:,:,0:(dataset.n_changepoints-1)])**2)
    
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
    
    
    
    # This evaluates the value of the integral for the rate function, across all pairs of nodes and timeframes
    # integral = 0
    # for k in list(range(dataset.n_changepoints)[0:(n_changepoints-1)]):
    #     Z_cur = Z[:,:,k]
    #     Z_new = Z[:,:,k+1]
    #     Sij00 = torch.mm(Z_cur,Z_cur.t())
    #     Sij01 = torch.mm(Z_cur,Z_new.t())
    #     Sij10 = torch.mm(Z_new,Z_cur.t())
    #     Sij11 = torch.mm(Z_new,Z_new.t())
    #     Sii00 = torch.sum(Z_cur*Z_cur,1).repeat(dataset.n_nodes,1)
    #     Sii01 = torch.sum(Z_cur*Z_new,1).repeat(dataset.n_nodes,1)
    #     Sii11 = torch.sum(Z_new*Z_new,1).repeat(dataset.n_nodes,1)
    #     S_mat = (Sij01 + Sij10 - Sii01 - Sii01.t()) / 3 + (2*Sij00 - Sii00 - Sii00.t()) / 3 + (2*Sij11 - Sii11 - Sii11.t()) / 3
    #     S_mat = S_mat.exp()
    #     integral += S_mat.triu(diagonal = 1).sum()
    
    # Integral of the rate function: exact value - no Jensen
    integral = 0.
    for k in list(range(dataset.n_changepoints)[0:(n_changepoints-1)]):
        tau_cur = dataset.changepoints[k]
        tau_new = dataset.changepoints[k+1]
        Z_cur = Z[:,:,k]
        Z_new = Z[:,:,k+1]
        tZ1  =  torch.sum(Z_cur**2, 1)
        tZ2 = tZ1.expand(tZ1.shape[0], tZ1.shape[0])
        tZ1 = tZ2.transpose(0,1)
        D = tZ1 + tZ2 -2*torch.mm(Z_cur, Z_cur.transpose(0,1))  # its element (i,j) is || z_i - z_j ||_2^2
        N = len(D)
        D_vec = D[torch.triu(torch.ones(N,N),1) == 1]
        DZ = Z_new - Z_cur            # This is \Delta 
        tDZ1 = torch.sum(DZ**2, 1)    
        tDZ2 = tDZ1.expand(tDZ1.shape[0], tDZ1.shape[0])
        tDZ1 = tDZ2.transpose(0,1)
        S = tDZ1 + tDZ2 -2*torch.mm(DZ, DZ.transpose(0,1))   # its element  (i,j) is || \Delta_i - \Delta_j ||_2^2
        S_vec = S[torch.triu(torch.ones(N,N),1) == 1]
        tA = torch.sum((DZ*Z_cur), 1)
        tB = tA.expand(tA.shape[0], tA.shape[0])
        tA = tB.transpose(0,1)
        C = torch.mm(DZ, Z_cur.transpose(0,1)) +  torch.mm(Z_cur, DZ.transpose(0,1)) - tA - tB
        C_vec = C[torch.triu(torch.ones(N,N),1) == 1]        
        mu = (1/S_vec)*(C_vec) 
        sigma = torch.sqrt(1/(2*S_vec))
        S = torch.exp(-(D_vec - S_vec*(mu**2)))*sigma*(RV.cdf((1 - mu)/sigma) - RV.cdf((0 - mu)/sigma))*(tau_new - tau_cur)
        integral += S.sum()        
    return prior - beta*len(dataset) - torch.sum(first_likelihood_term) + torch.sqrt(2*torch.tensor([math.pi], dtype=torch.float64))*torch.exp(beta)*integral

def FitOneShot(dataset, Z, optimiser, scheduler=None):
    optimiser.zero_grad()
    loss_function = clpm_negloglike(dataset, Z)
    loss_function.backward()
    optimiser.step()
    if  not scheduler == None:
        scheduler.step()
    return loss_function


### DATA AND PARAMETER INITIALISATION

edgelist = pd.read_csv('input/edgelist.csv')
changepoints = torch.tensor(pd.read_csv('input/changepoints.csv', header = None).values, dtype = torch.float64, device = device)
n_changepoints = len(changepoints)
timestamps = torch.tensor(edgelist.iloc[:,0:1].values, dtype = torch.float64, device = device)
interactions = torch.tensor(edgelist.iloc[:,1:3].values-1, dtype = torch.long, device = device)
n_nodes = torch.max(interactions).item() + 1
dataset = MDataset(timestamps, interactions, changepoints, transform = True)
Z = torch.tensor(np.random.normal(size = (n_nodes,2,n_changepoints)), dtype = torch.float64, device = device, requires_grad = True)
beta = torch.tensor(np.random.normal(size = 1), dtype = torch.float64, device = device, requires_grad = True) # intercept  term


#### DEBUG

#print("\n\n\n")
#clpm_negloglike(dataset, Z)
#print("\n\n\n")


### OPTIMISATION
epochs = 500
learning_rate = 1e-3
optimiser = torch.optim.SGD([beta, Z], lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size = epochs // 10, gamma = 0.35)
loss_function_values = np.zeros(epochs)
for epoch in range(epochs):
    loss_function_values[epoch] = FitOneShot(dataset, Z, optimiser).item()
    print("Epoch:", epoch, "\t\tLearning rate:", "{:2e}".format(optimiser.param_groups[0]['lr']), "\t\tLoss:", round(loss_function_values[epoch],3))


### EXPORT OUTPUT

Z_long_format = np.zeros([Z.shape[0]*Z.shape[1]*Z.shape[2], 4])# just writing the contents of this array in a long format so that I can read it with R
index = 0
for i in range(Z.shape[0]):
	for d in range(Z.shape[1]):
		for t in range(Z.shape[2]):
			Z_long_format[index, 0] = i+1
			Z_long_format[index, 1] = d+1
			Z_long_format[index, 2] = t+1
			Z_long_format[index, 3] = Z[i,d,t]
			index += 1

pd.DataFrame(Z_long_format).to_csv('output/positions.csv', index = False, header = False)
pd.DataFrame(loss_function_values).to_csv('output/loss_function_values.csv', index = False, header = False)


