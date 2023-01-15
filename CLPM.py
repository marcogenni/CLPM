#!/usr/bin/env python3

import math
import numpy as np
import pandas as pd
import torch
import time
import os
import sys
from torch.distributions.normal import Normal
RV = Normal(0, 1)
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from matplotlib import cm
cividis = cm.get_cmap('cividis', 12)
from datetime import datetime
from sklearn.cluster import SpectralClustering

sys.path.append('../')
from CLPM_dataset import *
from utils import *

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
        
        self.dataset = dataset
        self.model_type = model_type
        self.penalty = penalty
        self.n_change_points = n_change_points
        self.verbose = verbose

        self.n_epochs = 0
        self.batch_size = 0
        self.lr_z = 0
        self.lr_beta = 0
        self.grad_clip_value = 1e+100
        self.loss_values = torch.zeros(1, dtype=torch.float64, device=self.device)

        if model_type != 'distance' and model_type != 'projection':
            quit('ERROR: model_type is not supported')

        self.n_nodes = dataset.n_nodes
        self.time_max = max(dataset.timestamps).item()
        my_step = 1 / n_change_points
        self.change_points = np.arange(start=0.0, stop=1.0 + my_step,  step=my_step) * (self.time_max + 0.0001)
        self.change_points = torch.tensor(self.change_points, dtype=torch.float64, device=self.device)
        self.change_points.reshape(-1, 1)
        self.n_change_points = len(self.change_points)
        self.segment_length = self.change_points[1] - self.change_points[0]
        self.Z = torch.tensor(np.random.normal(size = (dataset.n_nodes, 2, self.n_change_points)), requires_grad = True, dtype=torch.float64, device=self.device)
        self.beta = torch.tensor(np.random.normal(), requires_grad = True, dtype=torch.float64, device=self.device)
        self.elapsed_secs = 0.
        self.last_epoch = 0

        # Predeclarations of plotting and video parameters from here onwards
        self.plt_coloring_method = 'degree'
        self.plt_coloring_n_groups = 1
        self.plt_dpi = 250
        self.plt_period = 1
        self.plt_size = (1200, 900)
        self.plt_frames_btw = 1
        self.plt_nodes_to_track = None
        self.plt_sub_graph = False
        self.plt_type_of = 'friendship'
        self.plt_n_hubs = 2
        self.plt_n_sub_nodes = 100
        self.plt_start_date = None
        self.plt_end_date = None
        self.plt_time_format = '%Y/%m/%d %H:%M:%S'

        self.node_colors = torch.zeros((self.n_nodes, self.n_change_points), dtype=torch.float64, requires_grad=False)
        self.node_sizes = torch.ones((self.n_nodes, self.n_change_points), dtype=torch.float64, requires_grad=False)

        self.plt_n_frames = None
        self.plt_change_points = None
        self.plt_Z = None
        self.plt_node_colors = None
        self.plt_node_sizes = None
        self.plt_Z_limit = None
        self.plt_images = []
        self.plt_datetimes = None

        if self.verbose is True:
            print(f'\nCLPM {self.model_type} model has been successfully initialized\n')

    def forward(self, dataset, nodes):
        """
        Calculates the negative log-likelihood function associated to the model
        @param      dataset             CLPM dataset object
        @param      nodes               list of labels indicating which nodes should be considered in this calculation
        """
        if self.model_type == 'projection':
            return self.projection_model_negloglike(dataset, nodes)
        if self.model_type == 'distance':
            return self.distance_model_negloglike(dataset, nodes)

    def loss(self, dataset, nodes):
        """
        simply renames the negative log-likelihood as the "loss"
        @param      dataset             CLPM dataset object
        @param      nodes               list of labels indicating which nodes should be considered in this calculation
        """
        return self.forward(dataset, nodes)

    def projection_model_negloglike(self, dataset, nodes):
        """
        Objective function for the projection CLPM - this corresponds to the negative penalized log-likelihood of the model
        @param      dataset             CLPM dataset object
        @param      nodes               list of labels indicating which nodes should be considered in this update
        """
        bs = len(nodes)
        fs = len(dataset)
        timestamps, senders, receivers = dataset[nodes]
        n_entries = len(timestamps)
        if n_entries <= 1:
            quit('ERROR: the batch only has one or less timestamped interactions. The current version of the code does not support this. Choose a larger batch_size.')

        prior = self.penalty * torch.sum(torch.sqrt((self.Z[nodes, 0, 1:self.n_change_points] - self.Z[nodes, 0, 0:(self.n_change_points - 1)])**2 + (self.Z[nodes, 1, 1:self.n_change_points] - self.Z[nodes, 1, 0:(self.n_change_points - 1)])**2))

        # This evaluates the poisson log-rates at the timestamps when each of the interactions happen
        kappa = (timestamps // self.segment_length).long()
        deltas = (timestamps / self.segment_length - kappa).squeeze()
        one_minus_deltas = torch.add(-deltas, 1).squeeze()
        Z_sender_cur = self.Z[senders, :, kappa].squeeze()
        Z_sender_new = self.Z[senders, :, kappa + 1].squeeze()
        Z_receiv_cur = self.Z[receivers, :, kappa].squeeze()
        Z_receiv_new = self.Z[receivers, :, kappa + 1].squeeze()
        first_likelihood_term_log = torch.zeros(n_entries, dtype=torch.float64, device=self.device)
        first_likelihood_term_log += self.beta
        first_likelihood_term_log += one_minus_deltas ** 2 * torch.sum(Z_sender_cur * Z_receiv_cur, 1)
        first_likelihood_term_log += deltas * one_minus_deltas * torch.sum(Z_sender_cur * Z_receiv_new, 1)
        first_likelihood_term_log += deltas * one_minus_deltas * torch.sum(Z_sender_new * Z_receiv_cur, 1)
        first_likelihood_term_log += deltas ** 2 * torch.sum(Z_sender_new * Z_receiv_new, 1)

        integral = 0.
        senders = torch.unique(torch.as_tensor(nodes, dtype=torch.long, device = self.device))
        receivers = torch.arange(fs)
        for k in list(range(self.n_change_points)[0:(self.n_change_points-1)]):
            tau_cur = self.change_points[k]
            tau_new = self.change_points[k + 1]
            Z_senders_cur = self.Z[senders, :, k]
            Z_receivers_cur = self.Z[receivers, :, k]
            Z_senders_new = self.Z[senders, :, (k+1)]
            Z_receivers_new = self.Z[receivers, :, (k+1)]
            a = torch.mm(Z_senders_new, Z_receivers_new.t())
            b = torch.mm(Z_senders_cur, Z_receivers_new.t()) + torch.mm(Z_senders_new, Z_receivers_cur.t())
            c = torch.mm(Z_senders_cur, Z_receivers_cur.t())
            approx_integral = (torch.ones(fs) - torch.eye(fs))[senders, :] * integrate_simpsons(a, b, c, 64) / 2
            integral += torch.exp(self.beta) * (tau_new - tau_cur) * torch.sum(approx_integral)
        return fs/bs*(prior - torch.sum(first_likelihood_term_log) + integral)

    def distance_model_negloglike(self, dataset, nodes):
        """
        Objective function for the distance CLPM - this corresponds to the negative penalized log-likelihood of the model
        @param      dataset             CLPM dataset object
        @param      nodes               list of labels indicating which nodes should be considered in this update
        """
        bs = len(nodes)
        fs = len(dataset)
        timestamps, senders, receivers = dataset[nodes]
        n_entries = len(timestamps)
        if n_entries <= 1:
            quit('ERROR: the batch only has one or less timestamped interactions. The current version of the code does not support this. Choose a larger batch_size.')

        # Prior contribution, this roughly corresponds to a gaussian prior on the initial positions and increments - you can think of this as a penalisation term
        prior = 0
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
            tZ2 = torch.sum(Z_receivers_cur**2, 1).reshape(-1,1)
            tZ2 = tZ2.expand(tZ2.shape[0], len(senders))
            tZ2 = tZ2.transpose(0, 1)            
            D = tZ1 + tZ2 - 2 * torch.mm(Z_senders_cur, Z_receivers_cur.transpose(0, 1))  # its element (i,j) is || z_i - z_j ||_2^2
            D_vec = D.flatten()
            zero_out_D = [D_vec>0.]
            D_vec = D_vec[zero_out_D]

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
            
        return fs/bs*(prior - self.beta * n_entries - torch.sum(first_likelihood_term) + torch.sqrt(2 * torch.tensor([math.pi], dtype=torch.float64, device = self.device)) * torch.exp(self.beta) * integral)

    def fit(self, dataset, n_epochs, batch_size, lr_z=1e-6, lr_beta=1e-7, grad_clip_value = 1e+100):
        """
        Runs the optimizer
        @param      dataset             CLPM dataset object
        @param      n_epochs            number of iterations for the optimization
        @param      batch_size          size of batch to consider in the stochastic gradient procedure
        @param      lr_z                learning rate for the latent positions
        @param      lr_beta             learning rate for the intercept parameter
        @param      grad_clip_value     gradient clipping threshold, in order to avoid exploding gradients
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
        self.grad_clip_value = grad_clip_value
        optimizer = torch.optim.SGD([{'params': self.Z}], lr=self.lr_z)
        if self.model_type == 'distance':
            optimizer = torch.optim.SGD([{'params': self.Z}, {'params': self.beta, 'lr': lr_beta}], lr=self.lr_z)

        # Optimization
        self.loss_values = torch.zeros(n_epochs, dtype=torch.float64)
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
                torch.nn.utils.clip_grad_value_(self.Z, self.grad_clip_value)
                optimizer.step()
                self.loss_values[epoch] += 1/len(batch_sizes) * loss.item()
                current_index += batch
                if self.verbose is True:
                    stringa = f'Elapsed seconds: {time.time()-start_time:.2f} \t\t Epoch: {epoch+1:>d} of {self.n_epochs:>d} \t\t Batch: {current_index:>d}/{len(dataset):>d}'
                    if current_index == len(dataset):
                        stringa += f'\t\t Loss: {self.loss_values[epoch]:>.4f}'
                    print(stringa)
                if torch.isnan(self.loss_values[epoch]).item():
                    quit('ERROR: loss is nan - reduce learning rate')
        self.elapsed_secs = time.time()-start_time
        self.last_epoch = epoch    
        print('\nOptimization has now finished.')
        print(f'\nThe optimal objective function value (based on the full dataset) is {self.loss(dataset, list(range(0,len(dataset)))).item():.4f}\n\n')

    def export_fit(self):
        """
        Exports the model and optimization parameters into a suitable folder
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
        plt.savefig(path + 'output_' + self.model_type + '/loss.pdf')
        plt.close()

        Z_long_format = np.zeros([self.Z.shape[0] * self.Z.shape[1] * self.Z.shape[2], 4])
        index = 0
        for i in range(self.Z.shape[0]):
            for d in range(self.Z.shape[1]):
                for t in range(self.Z.shape[2]):
                    Z_long_format[index, 0] = i
                    Z_long_format[index, 1] = d
                    Z_long_format[index, 2] = t
                    Z_long_format[index, 3] = self.Z[i, d, t].cpu()
                    index += 1
        
        pd.DataFrame(Z_long_format).to_csv(path + 'output_' + self.model_type + '/positions.csv', index=False, header=False)
        pd.DataFrame(self.loss_values).to_csv(path + 'output_' + self.model_type + '/loss_function_values.csv', index=False, header=False)
        pd.DataFrame(self.beta.cpu().detach().numpy().reshape(-1,1)).to_csv(path + 'output_' + self.model_type + '/beta.csv', index=False, header=False)

        np.savetxt(path + "output_" + self.model_type + "/changepoints.csv", self.change_points.cpu(), delimiter=',')
        np.savetxt(path + "output_" + self.model_type + "/elapsed_secs.csv", np.array(self.elapsed_secs).reshape(1,), delimiter = ",")
        np.savetxt(path + "output_" + self.model_type + "/n_epochs.csv", np.array(self.last_epoch).reshape(1,), delimiter = ",")

    def import_fit(self, folder=''):
        self.beta = pd.read_csv(folder + 'output_' + self.model_type + '/beta.csv', header=None).iloc[0, 0]
        self.Z = torch.zeros(size=(self.n_nodes, 2, self.n_change_points), dtype=torch.float64)
        Z_long = pd.read_csv(folder + 'output_' + self.model_type + '/positions.csv', header=None)
        for row in range(len(Z_long)):
            i = Z_long.iloc[row, 0]
            d = Z_long.iloc[row, 1]
            t = Z_long.iloc[row, 2]
            val = Z_long.iloc[row, 3]
            self.Z[i.astype('int'), d.astype('int'), t.astype('int')] = val

    def simu(self, nodes = None):
        if nodes == None:
            nodes = np.arange(len(self.Z))
        N = len(nodes)
        K = self.n_change_points
        adj_tensor = torch.zeros((N,N,K-1), dtype = torch.float32)

        for k in list(range(self.n_change_points)[0:(self.n_change_points - 1)]):
            tau_cur = self.change_points[k]
            tau_new = self.change_points[k + 1]
            Z_cur = self.Z[nodes, :, k]
            Z_new = self.Z[nodes, :, k + 1]
            tZ1 = torch.sum(Z_cur ** 2, 1)
            tZ2 = tZ1.expand(tZ1.shape[0], tZ1.shape[0])
            tZ1 = tZ2.transpose(0, 1)
            D = tZ1 + tZ2 - 2 * torch.mm(Z_cur, Z_cur.transpose(0, 1))  # its element (i,j) is || z_i - z_j ||_2^2
            # N = len(D)
            D_vec = D[torch.triu(torch.ones(N, N), 1) == 1]
            DZ = Z_new - Z_cur  # This is \Delta
            tDZ1 = torch.sum(DZ ** 2, 1)
            tDZ2 = tDZ1.expand(tDZ1.shape[0], tDZ1.shape[0])
            tDZ1 = tDZ2.transpose(0, 1)
            S = tDZ1 + tDZ2 - 2 * torch.mm(DZ, DZ.transpose(0, 1))  # its element  (i,j) is || \Delta_i - \Delta_j ||_2^2
            S_vec = S[torch.triu(torch.ones(N, N), 1) == 1]
            tA = torch.sum((DZ * Z_cur), 1)
            tB = tA.expand(tA.shape[0], tA.shape[0])
            tA = tB.transpose(0, 1)
            C = torch.mm(DZ, Z_cur.transpose(0, 1)) + torch.mm(Z_cur, DZ.transpose(0, 1)) - tA - tB
            C_vec = C[torch.triu(torch.ones(N, N), 1) == 1]
            mu = (1 / S_vec) * (C_vec)
            sigma = torch.sqrt(1 / (2 * S_vec))
            S = torch.sqrt(2*torch.tensor(math.pi, dtype = torch.float32))*torch.exp(self.beta)*torch.exp(-(D_vec - S_vec * (mu ** 2))) * sigma * (RV.cdf((1 - mu) / sigma) - RV.cdf((0 - mu) / sigma)) * (tau_new - tau_cur)
            counts = torch.poisson(S).float()
            tril_indices = torch.tril_indices(row=N, col=N, offset=-1)
            adj_tensor[:,:,k][tril_indices[0], tril_indices[1]] = counts
            adj_tensor[:,:,k] = adj_tensor[:,:,k] + adj_tensor[:,:,k].transpose(1,0)
            #integral += S.sum()
        return adj_tensor
        
    def def_plot_pars(self, options):
        """
        Sets the parameters that are used for creating plots and videos.
        @param      options     A dictionary containing the values for the parameters that you wish to set. Watch out for typos in your dict keys names because you wouldn't get an error.
        Options parameters:
        coloring_method         Can be one of "degree" (colors reflect sociability), "distance" (colors reflect distance from the center of the space), or "cluster" (colors refer to a partition obtained with spectral clustering from average pairwise distances).
        coloring_n_groups       If "coloring_method" is "cluster", then this indicates the number of groups in the partition
        dpi                     Image quality for the snapshots
        period                  For the video, this is the number of seconds that pass between two changepoints
        size                    Size of the image for snapshots
        frames_btw              Number of intermediate video frames that are inserted in-between two changepoints
        nodes_to_track          List of the labels of nodes that you would like to color differently in the plots, maximum of 8 nodes can be chosen
        sub_graph
        type_of
        n_hubs
        n_sub_nodes
        time_format             how you would like to visualize the time on each latent space snapshot
        start_date              datetime for the time corresponding to the first changepoint
        end_date                datetime for the time corresponding to the last changepoint
        """
        if options.get("coloring_method") is not None: self.plt_coloring_method = options["coloring_method"]
        if options.get("coloring_n_groups") is not None: self.plt_coloring_n_groups = options["coloring_n_groups"]
        if options.get("dpi") is not None: self.plt_dpi = options["dpi"]
        if options.get("period") is not None: self.plt_period = options["period"]
        if options.get("size") is not None: self.plt_size = options["size"]
        if options.get("frames_btw") is not None: self.plt_frames_btw = options["frames_btw"]
        if options.get("nodes_to_track") is not None: self.plt_nodes_to_track = options["nodes_to_track"]
        if options.get("sub_graph") is not None: self.plt_sub_graph = options["sub_graph"]
        if options.get("type_of") is not None: self.plt_type_of = options["type_of"]
        if options.get("n_hubs") is not None: self.plt_n_hubs = options["n_hubs"]
        if options.get("n_sub_nodes") is not None: self.plt_n_sub_nodes = options["n_sub_nodes"]
        if options.get("time_format") is not None: self.plt_time_format = options["time_format"]
        if options.get("start_date") is not None: self.plt_start_date = options["start_date"]
        if options.get("end_date") is not None: self.plt_end_date = options["end_date"]

    def fade_node_distance(self, bending=1):
        """
        Defines for each node at each time a value between 0 and 1 representing distance from the origin
        @param      bending         the values are amplified or reduced according to this positive value: >1 will increase variability, <1 will decrease variability
        """
        values = torch.zeros((self.n_nodes, self.n_change_points), dtype=torch.float64)
        for t in range(self.n_change_points):
            cZ = self.Z[:, :, t]
            tDZ1 = torch.sum(cZ ** 2, 1)
            tDZ2 = tDZ1.expand(tDZ1.shape[0], tDZ1.shape[0])
            tDZ1 = tDZ2.transpose(0, 1)
            S = (1.0 + tDZ1 + tDZ2 - 2 * torch.mm(cZ, cZ.transpose(0, 1)))
            S[range(len(S)), range(len(S))] = torch.zeros(len(S), dtype=torch.float64)
            values[:, t] = torch.mean(S, 1)
        values = values ** bending
        values /= values.min()
        values -= 1
        values /= values.max()  # now we have values between 0 and 1
        values *= -1
        values += 1  # now we have 1 minus the previous values
        return values.detach().numpy()

    def fade_node_degree(self, bending=1):
        """
        Defines for each node at each time a value between 0 and 1 representing sociability, represented by the total number of interactions in a given time interval
        @param      bending         the values are amplified or reduced according to this positive value: >1 will increase variability, <1 will decrease variability
        """
        values = torch.ones((self.n_nodes, self.n_change_points), dtype=torch.float64)
        for edge in range(self.dataset.n_entries):
            i = self.dataset.senders[edge].item()
            j = self.dataset.receivers[edge].item()
            kappa = (self.dataset.timestamps[edge] // self.segment_length).int().item()
            values[i, kappa] += 1
            values[j, kappa] += 1
            values[i, kappa+1] += 1
            values[j, kappa+1] += 1
        values = values ** bending
        values /= values.min()
        values -= 1
        values /= values.max()
        return values.detach().numpy()

    def expand_over_frames(self):
        """
        Takes a fitted CLPM, and fills-in intermediate values that are positioned in-between two changepoints.
        This is necessary in order to obtain smoother transitions in the video since more frames are added.
        """
        self.plt_n_frames = self.plt_frames_btw * (self.n_change_points - 1) + self.n_change_points
        self.plt_change_points = np.zeros(self.plt_n_frames)
        self.plt_change_points[self.plt_n_frames - 1] = self.change_points[self.n_change_points - 1]
        self.plt_Z = torch.zeros((self.n_nodes, 2, self.plt_n_frames), dtype = torch.float64)
        self.plt_Z[:, :, self.plt_n_frames - 1] = self.Z[:, :, self.n_change_points - 1]
        if self.plt_coloring_method != 'cluster':
            self.plt_node_colors = np.zeros((self.n_nodes, self.plt_n_frames))
            self.plt_node_colors[:, self.plt_n_frames - 1] = self.node_colors[:, self.n_change_points - 1]
        self.plt_node_sizes = np.zeros((self.n_nodes, self.plt_n_frames))
        self.plt_node_sizes[:, self.plt_n_frames - 1] = self.node_sizes[:, self.n_change_points - 1]
        for frame in range(self.plt_n_frames - 1):
            cp0 = frame // (self.plt_frames_btw + 1)
            cp1 = (frame // (self.plt_frames_btw + 1)) + 1
            delta = (frame % (self.plt_frames_btw + 1)) / (self.plt_frames_btw + 1)
            self.plt_change_points[frame] = (1 - delta) * self.change_points[cp0] + delta * self.change_points[cp1]
            for i in range(self.n_nodes):
                self.plt_Z[i, 0, frame] = (1 - delta) * self.Z[i, 0, cp0] + delta * self.Z[i, 0, cp1]
                self.plt_Z[i, 1, frame] = (1 - delta) * self.Z[i, 1, cp0] + delta * self.Z[i, 1, cp1]
                if self.plt_coloring_method != 'cluster': self.plt_node_colors[i, frame] = (1 - delta) * self.node_colors[i, cp0] + delta * self.node_colors[i, cp1]
                self.plt_node_sizes[i, frame] = (1 - delta) * self.node_sizes[i, cp0] + delta * self.node_sizes[i, cp1]
        self.plt_Z_limit = torch.abs(self.plt_Z).max().item() * 1.05
        self.plt_Z = self.plt_Z.detach().numpy()

    def interpolate_datetime_on_frames(self):
        """
        If self.plt_start_date and self.plt_end_date are specified by the user, then this function determines the date at each of the frames and stores it in self.plt_datetimes
        """
        if self.plt_start_date is not None:
            if self.plt_end_date is None:
                print('Fatal error: both start_date and end_date must be either of type None or tuples', sys.stderr)
            else:
                now = datetime(self.plt_start_date[0], self.plt_start_date[1], self.plt_start_date[2], self.plt_start_date[3], self.plt_start_date[4])
                last = datetime(self.plt_end_date[0], self.plt_end_date[1], self.plt_end_date[2], self.plt_end_date[3], self.plt_end_date[4])
                if now >= last:
                    print('Fatal error: end date is earlier than start date!', sys.stderr)
                delta = (last - now) / (self.plt_n_frames - 1)
                self.plt_datetimes = []
                while now <= last:
                    self.plt_datetimes.append(now.strftime(self.plt_time_format))
                    now += delta
                self.plt_datetimes.append(now.strftime(self.plt_time_format))

    def create_snaps(self, snapshots_only=False, bending_size=1, bending_colors=1):
        """
        Creates the latent space snapshots for all the required frames, and exports them into a results folder
        @param      snapshots_only     if True then the snapshots are created with trajectory lines, useful if you need to export and use images of single snapshots. If False, then the exported images correspond to those that will be used to form the video
        @param      bending_size       bending parameter used to determine the size of nodes in the plots
        @param      bending_colors     bending parameter used to determine the fading colors of nodes in the plots
        """
        self.node_sizes = self.fade_node_degree(bending_size)
        if self.plt_coloring_method == 'degree': self.node_colors = self.fade_node_degree(bending_colors)
        if self.plt_coloring_method == 'distance': self.node_colors = self.fade_node_distance(bending_colors)
        self.expand_over_frames()
        if self.plt_start_date is not None: self.interpolate_datetime_on_frames()
        special_colors = ['g', 'r', 'gold', 'b', 'c', 'm', 'y', 'k']
        cluster_labels = 0
        if self.plt_coloring_n_groups > len(special_colors): print("Error: maximum 8 groups allowed.", sys.stderr)
        if self.plt_coloring_method == 'cluster': cluster_labels = self.clustering()
        for frame in range(self.plt_n_frames):
            print('creating frame: ', frame + 1, ' out of ', self.plt_n_frames)
            plt.figure()
            if self.plt_datetimes is None:
                plt.title("Latent positions at " + str(round(self.plt_change_points[frame], 2)), loc="left")
            else:
                plt.title("Latent positions at " + self.plt_datetimes[frame], loc="left")
            plt.xlim((-self.plt_Z_limit, self.plt_Z_limit))
            plt.ylim((-self.plt_Z_limit, self.plt_Z_limit))
            if self.model_type == 'distance':
                plt.axhline(y=0, color='gray', linewidth=0.1)
                plt.axvline(x=0, color='gray', linewidth=0.1)
            else:
                for linea in np.arange(0, 2*np.pi, np.pi/4):
                    x_coord = [-2*self.plt_Z_limit * np.cos(linea), 2*self.plt_Z_limit * np.cos(linea)]
                    y_coord = [-2*self.plt_Z_limit * np.sin(linea), 2*self.plt_Z_limit * np.sin(linea)]
                    plt.plot(x_coord, y_coord, color='gray', linewidth=0.1)
                for circ in np.arange(0, 1.5*self.plt_Z_limit, 1.5*self.plt_Z_limit/5):
                    cerchio = plt.Circle((0, 0), circ, color='gray', linewidth=0.1, fill=False)
                    plt.gca().add_patch(cerchio)
            index_special_colors = 0
            for idi in range(self.n_nodes):
                this_nodes_color = 0
                if self.plt_coloring_method == 'cluster': this_nodes_color = special_colors[cluster_labels[idi]]
                else: this_nodes_color = cividis(self.plt_node_colors[idi, frame])
                if self.plt_nodes_to_track is not None:
                    if idi in self.plt_nodes_to_track:
                        this_nodes_color = special_colors[index_special_colors]
                        index_special_colors += 1
                if snapshots_only is False:
                    if frame >= 2: plt.plot([self.plt_Z[idi, 0, frame - 2], self.plt_Z[idi, 0, frame - 1]], [self.plt_Z[idi, 1, frame - 2], self.plt_Z[idi, 1, frame - 1]], '-', alpha=0.4, color=this_nodes_color)
                    if frame >= 1: plt.plot([self.plt_Z[idi, 0, frame - 1], self.plt_Z[idi, 0, frame - 0]], [self.plt_Z[idi, 1, frame - 1], self.plt_Z[idi, 1, frame - 0]], '-', alpha=0.6, color=this_nodes_color)
                plt.plot(self.plt_Z[idi, 0, frame], self.plt_Z[idi, 1, frame], 'o', markersize=5 + self.plt_node_sizes[idi, frame] * 7.5, markeredgewidth=0.2, alpha=0.8, markeredgecolor='black', markerfacecolor=this_nodes_color)
            folder_name = 'snaps'
            if snapshots_only is True:
                folder_name += '_only'
            plt.savefig('results_' + self.model_type + '/' + folder_name + '/snap_' + str(frame) + '.png', dpi=self.plt_dpi)
            plt.close()
        if snapshots_only is False:
            for i in range(self.plt_n_frames - 1):
                self.plt_images.append('results_' + self.model_type + '/snaps/snap_' + str(i) + '.png')

    def folder_management(self):
        """
        Prepares the folders to store the results and plots
        """
        list_files = os.listdir()

        if self.model_type == 'projection':
            if not 'output_projection' in list_files:
                print('fatal error: no output directory detected!', sys.stderr)
            if 'results_projection' in list_files:
                sub_list_files = os.listdir('results_projection/')
                if 'snaps' in sub_list_files:
                    if len(os.listdir('results_projection/snaps')) > 0:
                        for item in os.listdir('results_projection/snaps'):
                            if item.endswith('.png'):
                                os.remove(os.path.join('results_projection/snaps/', item))
                else:
                    os.mkdir('results_projection/snaps')
                if 'snaps_only' in sub_list_files:
                    if len(os.listdir('results_projection/snaps_only')) > 0:
                        for item in os.listdir('results_projection/snaps_only'):
                            if item.endswith('.png'):
                                os.remove(os.path.join('results_projection/snaps_only/', item))
                else:
                    os.mkdir('results_projection/snaps_only')
            if not 'results_projection' in list_files:
                os.mkdir('results_projection')
                os.mkdir('results_projection/snaps')
                os.mkdir('results_projection/snaps_only')

        if self.model_type == 'distance':
            if not 'output_distance' in list_files:
                print('fatal error: no output directory detected!', sys.stderr)
            if 'results_distance' in list_files:
                sub_list_files = os.listdir('results_distance/')
                if 'snaps' in sub_list_files:
                    if len(os.listdir('results_distance/snaps')) > 0:
                        for item in os.listdir('results_distance/snaps'):
                            if item.endswith('.png'):
                                os.remove(os.path.join('results_distance/snaps/', item))
                else:
                    os.mkdir('results_distance/snaps')
                if 'snaps_only' in sub_list_files:
                    if len(os.listdir('results_distance/snaps_only')) > 0:
                        for item in os.listdir('results_distance/snaps_only'):
                            if item.endswith('.png'):
                                os.remove(os.path.join('results_distance/snaps_only/', item))
                else:
                    os.mkdir('results_distance/snaps_only')
            if not 'results_distance' in list_files:
                os.mkdir('results_distance')
                os.mkdir('results_distance/snaps')
                os.mkdir('results_distance/snaps_only')

    def create_animation(self, extract_frames=False, bending_size=1, bending_colors=1):
        """
        Combines the functions create_snaps and make_video to produce the animation for a fitted clpm
        @param      extract_frames     whether you would like to also create an additional folder with the individual frames as standalone images
        @param      bending_size       bending parameter used to determine the size of nodes in the plots
        @param      bending_colors     bending parameter used to determine the fading colors of nodes in the plots
        """
        self.folder_management()
        if self.verbose:
            print("Folders are set up.")
            print("Creating snapshots...")
        self.create_snaps(False, bending_size, bending_colors)
        if extract_frames: self.create_snaps(True, bending_size, bending_colors)
        if self.verbose: print("Snapshots created.")
        fps = (self.plt_frames_btw + 1) / self.plt_period
        is_color = True
        format = "mp4v"
        if self.verbose: print("Creating video..")
        make_video('results_' + self.model_type + '/video.mp4', self.plt_images, fps, self.plt_size, is_color, format)
        if self.verbose: print("All done.")

    def clusteredness_index(self, thresholds, start_value, end_value, frames_btw):
        """
        Creates a csv and a plot for the clusteredness index
        @param      thresholds      threshold values for nearest neighbour mechanism (radius of circle around each node)
        @param      start_value     plotting parameter: lower bound for the x-axis labels
        @param      end_value       plotting parameter: upper bound for the x-axis labels
        @param      frames_btw      defines the number of coordinates that are used to draw the line in the plot, similarly to ClpmPlot()
        """
        n_thresh = len(thresholds)
        counts = np.zeros((self.plt_n_frames, n_thresh))
        timing = np.linspace(start_value, end_value, num=self.plt_n_frames)
        for frame in range(self.plt_n_frames):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        distance = math.sqrt((self.plt_Z[i, 0, frame] - self.plt_Z[j, 0, frame])**2 + (self.plt_Z[i, 1, frame] - self.plt_Z[j, 1, frame])**2)
                        for index in range(len(thresholds)):
                            if distance <= thresholds[index]:
                                counts[frame, index] += 1
        np.savetxt("results_distance/clusteredness.csv", counts/self.n_nodes, delimiter=',')
        plt.figure()
        plt.plot(timing[(frames_btw+1):], counts[(frames_btw+1):, :]/self.n_nodes)
        plt.xlabel("Time")
        plt.ylabel("Clusteredness")
        plt.legend(thresholds, title='threshold')
        plt.savefig("results_distance/clusteredness.pdf")
        plt.close()

    def clustering(self):
        """
        Calculates the median distance between all pairs of nodes. Uses a radial basis function to transform the distances into pairwise node similarities.
        Runs spectral clustering to find a partitioning of nodes. The number of groups is user-specified and read from self.plt_coloring_n_groups.
        """
        nodes_simil_all = np.zeros((self.n_nodes, self.n_nodes, self.plt_n_frames))
        for frame in range(self.plt_n_frames):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if i != j:
                        distance = math.sqrt((self.plt_Z[i, 0, frame] - self.plt_Z[j, 0, frame])**2 + (self.plt_Z[i, 1, frame] - self.plt_Z[j, 1, frame])**2)
                        nodes_simil_all[i, j, frame] = math.exp(-distance)
        nodes_simil_medians = np.ones((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    nodes_simil_medians[i, j] = np.median(nodes_simil_all[i, j, :])
        cluster_res = SpectralClustering(n_clusters=self.plt_coloring_n_groups, assign_labels='discretize', random_state=0, affinity='precomputed').fit(nodes_simil_medians)
        np.savetxt("output_" + self.model_type + "/cluster_labels.csv", cluster_res.labels_, delimiter=',')
        return cluster_res.labels_

    def reduce_network(self, edgelist, n_hubs=2, type_of='friendship', n_sub_nodes=100):
        sub_nodes, edgelist = get_sub_graph(edgelist.copy(), n_hubs=n_hubs, type_of=type_of, n_sub_nodes=n_sub_nodes)
        edgelist, conversion = edgelist_conversion(edgelist, sub_nodes, self.n_nodes)
        self.Z = self.Z[sub_nodes, :, :]
        self.dataset = NetworkCLPM(edgelist, self.verbose)
        self.n_nodes = self.dataset.n_nodes
        self.time_max = max(self.dataset.timestamps).item()
