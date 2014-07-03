# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:39:35 2014

@author: zhk
"""

import numpy as np
import scipy as sp
from sklearn import linear_model
import cPickle as pickle
import bitrade.research.util as util
import bitrade.research.convolution as convolution
import bitrade.research.simulation as simulation

def cluster(series, k, m, num_iter = 50, filter_level = 1):
    n = series.size
    samples = np.zeros((k, m))
    np.random.seed(seed=321)
    indices = np.random.random_integers(m,(n-1),k)
    for i in range(k):
        samples[i] = series[(indices[i]-m):(indices[i])]
    distance = np.zeros((k, n))
    prev_clusters = np.zeros(n)
    sequences = np.zeros((n,m))
    for i in range(m):
        sequences[:,i] = np.lib.pad(series, (m-i-1,0), 'constant', constant_values=(0, 0))[0:n]
    for iteration in range(num_iter):
        distance = convolution.measure_similarities(series, samples, False, num_processes = 10, filter_level = filter_level)
        clusters = np.argmax(distance, axis = 0)
        print iteration, np.count_nonzero(clusters - prev_clusters)
        prev_clusters = clusters
        for i in range(k):
            points = np.where((clusters==i)[m:])[0] + m
            subsequences = sequences[points]
            samples[i] = np.mean(subsequences, axis = 0)
    clusters[:m] = np.NaN
    for i in range(k):
        samples[i] = samples[i] - np.mean(samples[i])
    return clusters, samples, distance
    
def kmeans_fit(cluster_train, sample, distance_train, p1800_train, k, m, dimension):
    performance = np.zeros(k)
    for i in range(k):
        points = np.where(cluster_train==i)[0]
        performance[i] = np.mean(p1800_train[points])
    performance_rank = np.argsort(performance)
    best_sample = np.zeros((dimension, m))
    best_distance_train = np.zeros((dimension, len(distance_train[0])))
    for i in range(int(dimension/2)):
        best_sample[i] = sample[performance_rank[i]]
        best_sample[-i-1] = sample[performance_rank[-i-1]]
        best_distance_train[i] = distance_train[performance_rank[i]]
        best_distance_train[-i-1] = distance_train[performance_rank[-i-1]]        
    prob = np.exp(best_distance_train * 10) 
    prob = prob / np.sum(prob, axis = 0)
    prob = np.asmatrix(prob)
    coeff = prob * np.asmatrix(p1800_train).transpose() / np.sum(prob, axis = 1)
    return best_sample, coeff
    
def kmeans_predict(price_test, best_samples, coeff):
    best_distance_test = convolution.measure_similarities(price_test, best_samples, False, num_processes = 10, filter_level = 1)        
    prob = np.exp(best_distance_test * 10)
    prob = prob / np.sum(prob, axis = 0)
    prob = np.asmatrix(prob)
    _Y = prob.transpose() * coeff    
    _Y = np.array(_Y)[:, 0]
    return _Y


if __name__ == "__main__":
    exchange = 'okcoin'
    ticker   = 'btc_cny'
    start_time = 1392508800
    end_time   = 1403568000
    step = 10
    
    data, depth = util.load_data(start_time, end_time, exchange, ticker, step, True)
    data = pickle.load(open( "data.p", "rb" ))
    delay = 10
    delay_indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + 10)
    forward  = 300
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p300'] = depth.price.values[indices] - depth.price.values[delay_indices]
    forward  = 600
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p600'] = depth.price.values[indices] - depth.price.values[delay_indices]
    forward  = 900
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p900'] = depth.price.values[indices] - depth.price.values[delay_indices]
    forward  = 1200
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p1200'] = depth.price.values[indices] - depth.price.values[delay_indices]
    forward  = 1800
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p1800'] = depth.price.values[indices] - depth.price.values[delay_indices]
  
    
    train = data[:654000]
    test = data[681000:]
    k = 100
    dimension = 20
    price_train = train.price.values
    clusters_train = {}
    samples = {}
    distances_train = {}
    
    for m in [30, 60, 90, 120, 180, 360, 720]:
        clusters_train[str(m)], samples[str(m)], distances_train[str(m)] = cluster(price_train, k, m, num_iter = 50, filter_level = 1)

    for response in ['p300', 'p600', 'p900', 'p1200', 'p1800']:
        for m in [30, 60, 90, 120, 180, 360, 720]:
            best_samples, coeff = kmeans_fit(clusters_train[str(m)], samples[str(m)], distances_train[str(m)], train[response].values, k, m, dimension)
            _Y = kmeans_predict(data.price.values, best_samples, coeff)    
            Y = data[response].values
            data['s' + str(m) + '_' + response] = _Y
            print response, m, np.corrcoef(_Y[681000:], Y[681000:])[0,1], np.corrcoef(_Y[:654000:], Y[:654000])[0,1]
            #simulation.characterize_strategy(test, _Y[681000:], Y[681000:], 0.1, 0.6, 0.02, short_allowed = True, plot = False)
            #print
    
    response = 'p900'
    m = 360
    best_samples, coeff = kmeans_fit(clusters_train[str(m)], samples[str(m)], distances_train[str(m)], train[response].values, k, m, dimension)
    _Y = kmeans_predict(data.price.values, best_samples, coeff)    
    Y = data[response].values
    data['s' + str(m) + '_' + response] = _Y
    print response, m, np.corrcoef(_Y[681000:], Y[681000:])[0,1], np.corrcoef(_Y[:654000:], Y[:654000])[0,1]
    simulation.characterize_strategy(test, _Y[681000:], Y[681000:], 0.1, 0.6, 0.02, short_allowed = True, plot = False)