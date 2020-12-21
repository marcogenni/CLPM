#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  16 17:53:00 2020

@author: riccardo
"""

import math
import numpy as np
import pandas as pd

edgelist = pd.read_csv('dataset/ht09_contact_list.dat', sep = "\t", header = None)

edgelist.columns = ["timestamp", "sender", "receiver"]

edgelist.iloc[:,0] -= edgelist.iloc[:,0].min()
edgelist.iloc[:,0] /= edgelist.iloc[:,0].max()
edgelist.iloc[:,0] *= 24# (Marco: this should be changed to a suitable number of hours or minutes representing the length of the study)

edgelist.iloc[:,1] -= edgelist.iloc[:,1].min()

edgelist.iloc[:,2] -= edgelist.iloc[:,2].min()

edgelist = edgelist[["timestamp", "sender", "receiver"]]

n_nodes = max(edgelist.iloc[:,1:2].max()) + 1
permutation = np.zeros(n_nodes).astype(np.int64)

index = 1
for k in range(2):
    for l in range(edgelist.shape[0]):
        i = edgelist.iloc[l,k+1].astype(np.int64)
        if (permutation[i] == 0): 
            permutation[i] = index
            index += 1

for k in range(2):
    for l in range(edgelist.shape[0]):
        edgelist.iloc[l,k+1] = permutation[edgelist.iloc[l,k+1].astype(np.int64)] - 1

edgelist.to_csv("input/edgelist.csv", index = False, header = True)



