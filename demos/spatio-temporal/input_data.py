# -*- coding: utf-8 -*-
"""
Updated on  Wed Mar 11 09:28:50 2020

@author: 
"""

import numpy as np
import pandas as pd
import pickle as pkl

def load_sz_data(dataset):
    sz_adj = pd.read_csv(r'data/sz_adj.csv',header=None)
    adj = np.mat(sz_adj)
    sz_tf = pd.read_csv(r'data/sz_speed.csv')
    return sz_tf, adj

def load_los_data(dataset):
    los_adj = pd.read_csv(r'data/los_adj.csv',header=None)
    adj = np.mat(los_adj)
    los_tf = pd.read_csv(r'data/los_speed.csv')
    return los_tf, adj


def train_test_split(data, train_portion):
    time_len = data.shape[0]
    train_size = int(time_len * train_portion)
    train_data = np.array(data[:train_size])
    test_data  = np.array(data[train_size:])
    return train_data, test_data


def scale_data(train_data, test_data):
    max_speed = train_data.max()
    min_speed = train_data.min()
    train_scaled = (train_data - min_speed)/(max_speed - min_speed)
    test_scaled = (test_data - min_speed)/ (max_speed - min_speed)
    return train_scaled, test_scaled

def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []
    
    for i in range(len(train_data) - int(seq_len + pre_len -1)):
        a = train_data[i:i+seq_len+pre_len,]
        trainX.append(a[:seq_len])
        trainY.append(a[-1])
    
    for i in range(len(test_data) - int(seq_len + pre_len - 1)):
        b = test_data[i:i+seq_len+pre_len,]
        testX.append(b[:seq_len,])
        testY.append(b[-1])
    
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    
    return trainX, trainY, testX, testY