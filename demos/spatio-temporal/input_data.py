# -*- coding: utf-8 -*-
"""
Updated on  Wed Mar 11 09:28:50 2020

@author: 
"""

import numpy as np
import pandas as pd


class METR_LA(
    name="METR-LA",
    directory_name="METR-LA",
    url="https://github.com/lehaifeng/T-GCN/tree/master/data",
    url_archive_format="n/a",
    expected_files=[
        "los_speed.csv",
        "los_adj.csv",
    ],
    description="This traffic dataset contains traffic information collected from loop detectors in the highway of Los Angeles County (Jagadish et al., 2014).'",
    "There are several processed versions of this dataset used by the research community working in Traffic forecasting space.",
    "The data loaded here is  the pre-processed version of the dataset used by the TGCN paper.",
    "The data consists of two components:",
    "Time series: A N by D feature matrix, which describes the speed change over time on the roads.",
    "Graph structure: A N by N adjacency matrix, which describes the spatial relationship between roads",
    "The dataset includes 207 nodes (speed sensors) with 2016  speed records every 5 minutes for each sensor.",
    "Moreover, there is a distance proximity matrix of sensors for expressing their relationship.",
    source="https://github.com/lehaifeng/T-GCN/tree/master/data",
):
    def load(self,dataset):
        los_adj = pd.read_csv(r'data/los_adj.csv',header=None)
        adj = np.mat(los_adj)
        los_tf = pd.read_csv(r'data/los_speed.csv')
        return los_tf, adj
    
    
    def train_test_split(self, data, train_portion):
        time_len = data.shape[0]
        train_size = int(time_len * train_portion)
        train_data = np.array(data[:train_size])
        test_data  = np.array(data[train_size:])
        return train_data, test_data
    
    
    def scale_data(self, train_data, test_data):
        max_speed = train_data.max()
        min_speed = train_data.min()
        train_scaled = (train_data - min_speed)/(max_speed - min_speed)
        test_scaled = (test_data - min_speed)/ (max_speed - min_speed)
        return train_scaled, test_scaled
    
    def sequence_data_preparation(self, seq_len, pre_len, train_data, test_data):
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