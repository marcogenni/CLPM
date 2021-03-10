#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  16 17:53:00 2020

@author: riccardo
"""

import math
import numpy as np
import pandas as pd

edgelist = pd.read_csv('dataset/downloaded/out_in_better_format.csv', sep = "\t", header = None)

edgelist.iloc[:,0] -= 1
edgelist.iloc[:,1] -= 1

edgelist.iloc[:,3].min()
edgelist.iloc[:,3].max()

edgelist.iloc[:,3] = edgelist.iloc[:,3] - edgelist.iloc[:,3].min()
edgelist.iloc[:,3] = edgelist.iloc[:,3] / edgelist.iloc[:,3].max()
edgelist.iloc[:,3] = edgelist.iloc[:,3] * 233 # the study was 233 days long

edgelist = edgelist.iloc[:,[0,1,3]]

edgelist.columns = ["sender", "receiver", "timestamp"]

edgelist = edgelist[["timestamp", "sender", "receiver"]]

edgelist = edgelist[edgelist.timestamp < 10]

edgelist.to_csv("./edgelist.csv", index = False, header = True)

edgelist.iloc[:,1].min()
edgelist.iloc[:,1].max()

edgelist.iloc[:,2].min()
edgelist.iloc[:,2].max()


