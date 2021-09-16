#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import torch
import time
import os
import sys
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
RV = Normal(0,1)


class ModelCLPM(torch.nn.Module):
    def __init__(self,
                 dataset,
                 n_change_points,
                 model_type='distance',
                 penalty=10.,
                 verbose=True,
                 ):

        super().__init__()
        
        self.device = dataset.device
        
        self.model_type = model_type
        self.penalty = penalty
        self.n_change_points = n_change_points
        self.verbose = verbose

        self.n_epochs = 0
        self.batch_size = 0
        self.lr_z = 0
        self.lr_beta = 0
        self.loss_values = torch.zeros(1, dtype=torch.float64, device=self.device)
        # self.loss_values_debug = torch.zeros(1, dtype=torch.float64)         # DEBUG

        if model_type != 'distance' and model_type != 'projection':
            quit('ERROR: model_type is not supported')

        self.time_max = max(dataset.timestamps).item()
        my_step = 1 / n_change_points
        self.change_points = np.arange(start=0.0, stop=1.0 + my_step,  step=my_step) * (self.time_max + 0.0001)
        self.change_points = torch.tensor(self.change_points, dtype=torch.float64, device=self.device)
        self.change_points.reshape(-1, 1)
        self.n_change_points = len(self.change_points)
        self.segment_length = self.change_points[1] - self.change_points[0]
        self.Z = torch.tensor(np.random.normal(size = (dataset.n_nodes, 2, self.n_change_points)), requires_grad = True, dtype=torch.float64, device=self.device)
        if self.model_type == 'distance':
            self.beta = torch.tensor(np.random.normal(), requires_grad = True, dtype=torch.float64, device = self.device)
        if self.verbose is True:
            print(f'\nCLPM {self.model_type} model has been successfully initialized\n')

    def forward(self, dataset, nodes):
        """
        calculates the negative log-likelihood function associated to the model
        """
        if self.model_type == 'projection':
            return self.projection_model_negloglike(dataset, nodes)
        if self.model_type == 'distance':
            return self.distance_model_negloglike(dataset, nodes)

    def loss(self, dataset, nodes):
        """
        simply renames the negative log-likelihood as the "loss"
        """
        return self.forward(dataset, nodes)

    def projection_model_negloglike(self, dataset, nodes):
        """
        Objective function for the projection CLPM - this corresponds to the negative penalized log-likelihood of the model
        """
        bs = len(nodes)
        fs = len(dataset)
        timestamps, senders, receivers = dataset[nodes]
        n_entries = len(timestamps)
        if n_entries <= 1:
            quit('ERROR: the batch only has one or less timestamped interactions. The current version of the code does not support this. Choose a larger batch_size.')

        Z = torch.exp(self.Z)
        prior = 0.
        norm_Z = torch.sqrt(torch.sum(Z ** 2, 1))
        norm_Z = norm_Z.unsqueeze(1)
        norm_Z = norm_Z.expand_as(Z)
        scaled_Z = Z / norm_Z
        #prior += self.penalty * torch.mean((Z[nodes, :, 1:self.n_change_points] - Z[nodes, :, :-1]) ** 2)
        prior += self.penalty * torch.sum((torch.sum(scaled_Z[nodes, :, 1:self.n_change_points] * scaled_Z[nodes, :, :-1], 1) - 1.) ** 2)
        # prior = 0
        # prior += self.penalty * torch.sum(self.Z[nodes, :, 0] ** 2)
        # prior += self.penalty * torch.sum((self.Z[nodes, :, 1:self.n_change_points] - self.Z[nodes, :, 0:(self.n_change_points - 1)]) ** 2)

        # This evaluates the poisson log-rates at the timestamps when each of the interactions happen
        kappa = (timestamps // self.segment_length).long()
        deltas = (timestamps / self.segment_length - kappa).squeeze()
        one_minus_deltas = torch.add(-deltas, 1).squeeze()
        Z_sender_cur = Z[senders, :, kappa].squeeze()
        Z_sender_new = Z[senders, :, kappa + 1].squeeze()
        Z_receiv_cur = Z[receivers, :, kappa].squeeze()
        Z_receiv_new = Z[receivers, :, kappa + 1].squeeze()
        first_likelihood_term = torch.zeros(n_entries, dtype=torch.float64, device = self.device)
        first_likelihood_term += one_minus_deltas ** 2 * torch.sum(Z_sender_cur * Z_receiv_cur, 1)
        first_likelihood_term += deltas * one_minus_deltas * torch.sum(Z_sender_cur * Z_receiv_new, 1)
        first_likelihood_term += deltas * one_minus_deltas * torch.sum(Z_sender_new * Z_receiv_cur, 1)
        first_likelihood_term += deltas ** 2 * torch.sum(Z_sender_new * Z_receiv_new, 1)

        # This evaluates the value of the integral for the rate function, across all pairs of nodes and timeframes
        #######################
        #### First Version ####
        #######################        
                # integral = 0.
                # for k in list(range(self.n_change_points)[0:(self.n_change_points-1)]):
                #     tau_cur = self.change_points[k]
                #     tau_new = self.change_points[k + 1]
                #     Z_cur = Z[nodes, :, k]
                #     Z_new = Z[nodes, :, k + 1]
                #     Sij00 = (torch.sum(torch.mm(Z_cur, Z_cur.t())) - torch.sum(Z_cur * Z_cur)) / 6
                #     Sij11 = (torch.sum(torch.mm(Z_new, Z_new.t())) - torch.sum(Z_new * Z_new)) / 6
                #     Sij01 = (torch.sum(torch.mm(Z_cur, Z_new.t())) - torch.sum(Z_cur * Z_new)) / 12
                #     Sij10 = (torch.sum(torch.mm(Z_new, Z_cur.t())) - torch.sum(Z_new * Z_cur)) / 12
                #     integral += (tau_new - tau_cur) * (Sij00 + Sij11 + Sij01 + Sij10)
        
                # return prior - torch.sum(torch.log(first_likelihood_term)) + integral
        ##########################
        #### New Version : MB ####
        ##########################
        integral = 0.
        senders = torch.unique(senders)
        receivers = torch.arange(fs)
        for k in list(range(self.n_change_points)[0:(self.n_change_points-1)]):
            tau_cur = self.change_points[k]
            tau_new = self.change_points[k + 1]
            Z_senders_cur = Z[senders, : , k]
            Z_receivers_cur = Z[receivers, : , k]
            Z_senders_new = Z[senders, : , (k+1)]
            Z_receivers_new = Z[receivers, : , (k+1)]
            Sij00 = (torch.sum(torch.mm(Z_senders_cur, Z_receivers_cur.t())) - torch.sum(Z_senders_cur * Z_senders_cur)) / 6
            Sij11 = (torch.sum(torch.mm(Z_senders_new, Z_receivers_new.t())) - torch.sum(Z_senders_new * Z_senders_new)) / 6
            Sij01 = (torch.sum(torch.mm(Z_senders_cur, Z_receivers_new.t())) - torch.sum(Z_senders_cur * Z_senders_new)) / 12 
            Sij10 = (torch.sum(torch.mm(Z_senders_new, Z_receivers_cur.t())) - torch.sum(Z_senders_new * Z_senders_cur)) / 12            
            integral += (tau_new - tau_cur) * (Sij00 + Sij11 + Sij01 + Sij10)

        return fs/bs*(prior - torch.sum(torch.log(first_likelihood_term)) + integral)            

    def distance_model_negloglike(self, dataset, nodes):
        """
        Objective function for the distance CLPM - this corresponds to the negative penalized log-likelihood of the model
        """
        bs = len(nodes)
        fs = len(dataset)
        # print("Batch size: "+str(bs)+" Full size: "+str(fs))
        timestamps, senders, receivers = dataset[nodes]
        n_entries = len(timestamps)
        if n_entries <= 1:
            quit('ERROR: the batch only has one or less timestamped interactions. The current version of the code does not support this. Choose a larger batch_size.')

        # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
        prior = 0
        prior += self.penalty * torch.sum(self.Z[nodes, :, 0] ** 2)
        prior += self.penalty * torch.sum((self.Z[nodes, :, 1:self.n_change_points] - self.Z[nodes, :, 0:(self.n_change_points - 1)]) ** 2)

        # This evaluates the poisson logrates at the timestamps when each of the interactions happen
        kappa = (timestamps // self.segment_length).long()
        deltas = (timestamps / self.segment_length - kappa.double()).squeeze()
        one_minus_deltas = torch.add(-deltas, 1).squeeze()
        Z_sender_cur = self.Z[senders, :, kappa].squeeze()
        Z_sender_new = self.Z[senders, :, kappa + 1].squeeze()
        Z_receiv_cur = self.Z[receivers, :, kappa].squeeze()
        Z_receiv_new = self.Z[receivers, :, kappa + 1].squeeze()
        first_likelihood_term = torch.zeros(n_entries, dtype=torch.float64, device=self.device)
        first_likelihood_term += one_minus_deltas ** 2 * (2 * torch.sum(Z_sender_cur * Z_receiv_cur, 1) - torch.sum(Z_sender_cur * Z_sender_cur, 1) - torch.sum(Z_receiv_cur * Z_receiv_cur, 1))
        first_likelihood_term += 2 * deltas * one_minus_deltas * (
                    torch.sum(Z_sender_cur * Z_receiv_new, 1) + torch.sum(Z_sender_new * Z_receiv_cur, 1) - torch.sum(Z_sender_cur * Z_sender_new, 1) - torch.sum(Z_receiv_cur * Z_receiv_new, 1))
        first_likelihood_term += deltas ** 2 * (2 * torch.sum(Z_sender_new * Z_receiv_new, 1) - torch.sum(Z_sender_new * Z_sender_new, 1) - torch.sum(Z_receiv_new * Z_receiv_new, 1))
        #print("first ll term: ",torch.sum(first_likelihood_term))
        
        #########################################
        ##############  First version ###########
        #########################################
                # Integral of the rate function
                # integral = 0.
        
                # for k in list(range(self.n_change_points)[0:(self.n_change_points - 1)]):
                #     tau_cur = self.change_points[k]
                #     tau_new = self.change_points[k + 1]
                #     Z_cur = self.Z[nodes, :, k]
                #     Z_new = self.Z[nodes, :, k + 1]
                #     tZ1 = torch.sum(Z_cur ** 2, 1)
                #     tZ2 = tZ1.expand(tZ1.shape[0], tZ1.shape[0])
                #     tZ1 = tZ2.transpose(0, 1)
                #     D = tZ1 + tZ2 - 2 * torch.mm(Z_cur, Z_cur.transpose(0, 1))  # its element (i,j) is || z_i - z_j ||_2^2
                #     N = len(D)
                #     D_vec = D[torch.triu(torch.ones(N, N), 1) == 1]
                #     DZ = Z_new - Z_cur  # This is \Delta
                #     tDZ1 = torch.sum(DZ ** 2, 1)
                #     tDZ2 = tDZ1.expand(tDZ1.shape[0], tDZ1.shape[0])
                #     tDZ1 = tDZ2.transpose(0, 1)
                #     S = tDZ1 + tDZ2 - 2 * torch.mm(DZ, DZ.transpose(0, 1))  # its element  (i,j) is || \Delta_i - \Delta_j ||_2^2
                #     S_vec = S[torch.triu(torch.ones(N, N), 1) == 1]
                #     tA = torch.sum((DZ * Z_cur), 1)
                #     tB = tA.expand(tA.shape[0], tA.shape[0])
                #     tA = tB.transpose(0, 1)
                #     C = torch.mm(DZ, Z_cur.transpose(0, 1)) + torch.mm(Z_cur, DZ.transpose(0, 1)) - tA - tB
                #     C_vec = C[torch.triu(torch.ones(N, N), 1) == 1]
                #     mu = (1 / S_vec) * (C_vec)
                #     sigma = torch.sqrt(1 / (2 * S_vec))
                #     S = torch.exp(-(D_vec - S_vec * (mu ** 2))) * sigma * (RV.cdf((1 - mu) / sigma) - RV.cdf((0 - mu) / sigma)) * (tau_new - tau_cur)
                #     integral += S.sum()
                    
        ###############################################
        ##############  Second version (MB) ###########
        ###############################################
        # Integral of the rate function
        integral = 0.
        
        senders = torch.unique(senders)
        receivers = torch.arange(fs)
        for k in list(range(self.n_change_points)[0:(self.n_change_points - 1)]):
            tau_cur = self.change_points[k]
            tau_new = self.change_points[k+1]
            Z_senders_cur = self.Z[senders, :, k]
            Z_receivers_cur = self.Z[receivers, :, k]
            Z_senders_new = self.Z[senders, :, k + 1]
            Z_receivers_new = self.Z[receivers, :, k + 1]
            tZ1 = torch.sum(Z_senders_cur ** 2, 1).reshape(-1,1)
            tZ1 = tZ1.expand(tZ1.shape[0], len(receivers))
            #print("tZ1 shape: ", tZ1.shape)
            tZ2 = torch.sum(Z_receivers_cur**2, 1).reshape(-1,1)
            tZ2 = tZ2.expand(tZ2.shape[0], len(senders))
            tZ2 = tZ2.transpose(0, 1)            
            #print("tZ2 shape: ", tZ2.shape)
            D = tZ1 + tZ2 - 2 * torch.mm(Z_senders_cur, Z_receivers_cur.transpose(0, 1))  # its element (i,j) is || z_i - z_j ||_2^2
            D_vec = D.flatten()
            zero_out_D = [D_vec>0.]
            D_vec = D_vec[zero_out_D]
            #print("Guarda qui: ", D_vec)
            
            DZ_senders = Z_senders_new - Z_senders_cur  # This is \Delta (senders)
            DZ_receivers = Z_receivers_new - Z_receivers_cur  # This is \Delta (receivers)            
            tDZ1 = torch.sum(DZ_senders ** 2, 1).reshape(-1,1)
            tDZ1 = tDZ1.expand(tDZ1.shape[0], len(receivers))
            tDZ2 = torch.sum(DZ_receivers ** 2, 1).reshape(-1,1)
            tDZ2 = tDZ2.expand(tDZ2.shape[0], len(senders))
            tDZ2 = tDZ2.transpose(0, 1)
            S = tDZ1 + tDZ2 - 2 * torch.mm(DZ_senders, DZ_receivers.transpose(0, 1))  # its element  (i,j) is || \Delta_i - \Delta_j ||_2^2
            
            S_vec = S.flatten()
            S_vec = S_vec[zero_out_D]
            zero_out_S = [S_vec>0.]
            D_vec = D_vec[zero_out_S]
            S_vec = S_vec[zero_out_S]
            
            #print("E poi qui: ", S_vec)
            tA = torch.sum((DZ_senders * Z_senders_cur), 1).reshape(-1,1)
            tA = tA.expand(tA.shape[0], len(receivers))            
            tB = torch.sum(DZ_receivers * Z_receivers_cur, 1).reshape(-1,1)
            tB = tB.expand(tB.shape[0], len(senders))
            tB = tB.transpose(0, 1)
            C = torch.mm(DZ_senders, Z_receivers_cur.transpose(0, 1)) + torch.mm(Z_senders_cur, DZ_receivers.transpose(0, 1)) - tA - tB
            
            C_vec = C.flatten()
            C_vec = C_vec[zero_out_D]
            C_vec = C_vec[zero_out_S]
            mu = (1 / S_vec) * (C_vec)
            sigma = torch.sqrt(1 / (2 * S_vec))
            S = torch.exp(-(D_vec - S_vec * (mu ** 2))) * sigma * (RV.cdf((1 - mu) / sigma) - RV.cdf((0 - mu) / sigma)) * (tau_new - tau_cur)
            integral += .5*(S.sum()) # - bs*(tau_new - tau_cur)) # il secondo termine è per togliere i self-loops
            
        ############################
        ############################     
        #print("test S: ", S_vec[S_vec<0.])       
        #print("integral value: ", integral)   
        return fs/bs*(prior - self.beta * n_entries - torch.sum(first_likelihood_term) + torch.sqrt(2 * torch.tensor([math.pi], dtype=torch.float64, device = self.device)) * torch.exp(self.beta) * integral)

    def fit(self, dataset, n_epochs, batch_size, lr_z=1e-3, lr_beta=1e-7, threshold = None):
        """
        Runs the optimizer
        """
        # Batch size management
        self.batch_size = batch_size
        if self.batch_size < 1:
            quit('ERROR: batch_size is too small')
        if self.batch_size > len(dataset):
            quit('ERROR: batch_size cannot be larger than the number of nodes')
        n_batches = len(dataset)//self.batch_size
        batch_sizes = [self.batch_size] * n_batches
        remainder_nodes = len(dataset)- n_batches * self.batch_size
        if remainder_nodes > batch_size/2:
            batch_sizes.append(remainder_nodes)
            n_batches += 1
        if self.verbose is True:
            print("Batch sizes will be", batch_sizes, "covering", 100*sum(batch_sizes)/len(dataset), "percent of the dataset in each epoch")

        # Optimizer
        self.n_epochs = n_epochs
        self.lr_z = lr_z
        self.lr_beta = lr_beta
        optimizer = torch.optim.SGD([{'params': self.Z}], lr=self.lr_z)
        if self.model_type == 'distance':
            optimizer = torch.optim.SGD([{'params': self.Z}, {'params': self.beta, 'lr': lr_beta}], lr=self.lr_z)
        # for param_group in optimizer.param_groups:
        #     print(param_group)

        # Optimization
        self.loss_values = torch.zeros(n_epochs, dtype=torch.float64)
        # self.loss_values_debug = torch.zeros(n_epochs, dtype=torch.float64)         # DEBUG
        start_time = time.time()
        if self.verbose is True:
            print('\nStochastic gradient descent starting now:')
        for epoch in range(n_epochs):
            nodes_ordering = torch.randperm(dataset.n_nodes).tolist()
            current_index = 0
            if self.verbose is True:
                print("")
            for batch in batch_sizes:
                nodes = nodes_ordering[current_index:(current_index+batch)]
                loss = self.loss(dataset, nodes)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.loss_values[epoch] += 1/len(batch_sizes) * loss.item()
                current_index += batch
            if self.verbose is True:
                print(f'Elapsed seconds: {time.time()-start_time:.2f} \t\t Epoch: {epoch+1:>d} \t\t Batch: {current_index+batch:>d}/{len(dataset):>d} \t\t Loss: {self.loss_values[epoch]:>.4f}')
            if threshold!=None:
                if self.loss_values[epoch] < threshold:
                    # self.elapsed_secs = time.time()-start_time
                    # self.last_epoch = epoch
                    break
            # self.loss_values_debug[epoch] = self.loss(dataset, nodes_ordering).item()  # DEBUG
        self.elapsed_secs = time.time()-start_time
        self.last_epoch = epoch    
        print('\nOptimization has now finished.')
        print(f'\nThe optimal objective function value (based on the full dataset) is {self.loss(dataset, list(range(0,len(dataset)))).item():.4f}')

    def export(self):
        """
        Export the model and optimization parameters into a suitable folder
        """
        path = ""
        list_files = os.listdir()
        if 'output_projection' not in list_files:
            os.mkdir('output_projection')
        if 'output_distance' not in list_files:
            os.mkdir('output_distance')
        if 'results_projection' not in list_files:
            os.mkdir('results_projection')
            os.mkdir('results_projection/snaps')
        if 'results_distance' not in list_files:
            os.mkdir('results_distance')
            os.mkdir('results_distance/snaps')
            plt.figure()
        plt.plot(self.loss_values)
        # plt.plot(self.loss_values_debug)         # DEBUG
        # plt.plot(self.loss_values * (0.5 * self.Z.shape[0] * (self.Z.shape[0] - 1)) / (0.5 * self.batch_size * (self.batch_size - 1)))
        # plt.plot(self.loss_values * self.Z.shape[0] / self.batch_size)
        plt.savefig(path + 'output_' + self.model_type + '/loss.pdf')
        plt.close()

        Z_long_format = np.zeros([self.Z.shape[0] * self.Z.shape[1] * self.Z.shape[2], 4])
        index = 0
        for i in range(self.Z.shape[0]):
            for d in range(self.Z.shape[1]):
                for t in range(self.Z.shape[2]):
                    Z_long_format[index, 0] = i + 1
                    Z_long_format[index, 1] = d + 1
                    Z_long_format[index, 2] = t + 1
                    Z_long_format[index, 3] = self.Z[i, d, t].cpu()
                    index += 1
        
        pd.DataFrame(Z_long_format).to_csv(path + 'output_' + self.model_type + '/positions.csv', index=False, header=False)
        pd.DataFrame(self.loss_values).to_csv(path + 'output_' + self.model_type + '/loss_function_values.csv', index=False, header=False)
        if self.model_type == 'distance':
            pd.DataFrame(self.beta.cpu().detach().numpy().reshape(-1,1)).to_csv(path + 'output_' + self.model_type + '/beta.csv', index=False, header=False)

        np.savetxt(path + "output_" + self.model_type + "/changepoints.csv", self.change_points.cpu(), delimiter=',')
        np.savetxt(path + "output_" + self.model_type + "/elapsed_secs.csv", np.array(self.elapsed_secs).reshape(1,), delimiter = ",")
        np.savetxt(path + "output_" + self.model_type + "/n_epochs.csv", np.array(self.last_epoch).reshape(1,), delimiter = ",")

