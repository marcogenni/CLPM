#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:00:00 2020
@author: Marco and Riccardo
"""

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

def projection_model_negloglike(dataset, Z, device = "cpu"):
    # Make sure that we are dealing with nonnegative values only
    Z = Z.abs()
    
    # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
    prior = 0
    prior += 0.* torch.sum(Z[:,:,0]**2)
    prior += 7. * torch.sum((Z[:,:,1:(dataset.n_changepoints)] - Z[:,:,0:(dataset.n_changepoints-1)])**2)
    
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

def distance_negloglike(dataset, Z, beta, device = "cpu"):
    # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
    prior = 0
    prior += 10.0 * torch.sum(Z[:,:,0]**2)
    prior += 10.0 * torch.sum((Z[:,:,1:(dataset.n_changepoints)] - Z[:,:,0:(dataset.n_changepoints-1)])**2)
    
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

def FitOneShot(dataset, Z, beta, optimiser, scheduler=None, device = "cpu"):
    optimiser.zero_grad()
    loss_function = distance_negloglike(dataset, Z, beta, device)
    loss_function.backward()
    optimiser.step()
    if not scheduler == None:
        scheduler.step()
    return loss_function



