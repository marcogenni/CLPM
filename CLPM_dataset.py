#!/usr/bin/env python3


import torch
from torch.utils.data import Dataset


class NetworkCLPM(Dataset):
    def __init__(self, edge_list, verbose=True, device = 'cpu'):
        self.device = device
        self.timestamps = torch.tensor(edge_list.iloc[:, 0:1].values, dtype=torch.float64, device = device)
        # networks are undirected but we use names such as senders and receivers to easily identify the columns of the edge list
        self.senders = torch.tensor(edge_list.iloc[:, 1:2].values, dtype=torch.long, device = device)
        self.receivers = torch.tensor(edge_list.iloc[:, 2:3].values, dtype=torch.long, device = device)
        self.n_entries = len(self.timestamps)
        self.n_nodes = torch.max(self.senders).item() + 1
        if torch.max(self.receivers).item() + 1 > self.n_nodes:
            self.n_nodes = torch.max(self.receivers).item() + 1

        # we create a matrix such that for every pair i and j we know how many entries in the edge list refer to an interaction between i and j
        self.n_edges_per_pair = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.long, device = device)
        for edge in range(self.n_entries):
            self.n_edges_per_pair[self.senders[edge], self.receivers[edge]] += 1
            self.n_edges_per_pair[self.receivers[edge], self.senders[edge]] += 1

        # store the highest number of dyadic interactions
        self.most_interactions = torch.max(self.n_edges_per_pair).item() + 1

        # create a cube which indicates, for every pair (i, j), the row indices in the edge list for all the interactions between i and j
        self.adj_box = torch.zeros((self.n_nodes, self.n_nodes, self.most_interactions), dtype=torch.long, device = device)
        n_edges_per_pair_temp = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.long, device = device)
        for edge in range(self.n_entries):
            sender_index = self.senders[edge]
            receiver_index = self.receivers[edge]
            self.adj_box[sender_index, receiver_index, n_edges_per_pair_temp[sender_index, receiver_index]] = edge
            self.adj_box[receiver_index, sender_index, n_edges_per_pair_temp[receiver_index, sender_index]] = edge
            n_edges_per_pair_temp[sender_index, receiver_index] += 1
            n_edges_per_pair_temp[receiver_index, sender_index] += 1

        if verbose is True:
            print(f'\nLoaded network with {self.n_nodes} nodes and {self.n_entries} timestamped interactions')

    def __getitem__(self, item):
        """
        this function extracts the subnetwork which involves only the nodes provided in "item"
        """
        entries = []
        if isinstance(item, list):
            for i in item:
                for j in range(self.n_nodes):#item:
                    if i < j:
                        for entry_position in range(self.n_edges_per_pair[i, j]):
                            entries.append(self.adj_box[i, j, entry_position].item())
        else:
            quit("ERROR: can only extract subsets of NetworkCLPM if a list of nodes is provided")
        return [self.timestamps[entries], self.senders[entries], self.receivers[entries]]

    def __len__(self):
        """
        the "length" of the dataset is defined as the number of nodes in the dataset
        """
        return self.n_nodes
