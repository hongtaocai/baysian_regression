# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 03:14:07 2014

@author: zhk
"""

import numpy as np
import scipy as sp
from sklearn import linear_model
import bitrade.research.util as util
import bitrade.research.convolution as convolution
import bitrade.research.simulation as simulation

exchange = 'okcoin'
ticker   = 'btc_cny'
start_time = 1392508800
end_time   = 1402444800
step = 10

data, depth = util.load_data(start_time, end_time, exchange, ticker, step, False)
forward  = 1800
indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
data['p1800'] = -depth.price[data.depth_index].values + depth.price[indices].values

m = 360
k = 100
n = len(data)

def cluster_mixture(series, k, m, num_iter = 50, filter_level = 1):
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
    sequences = np.matrix(sequences)
    for iteration in range(num_iter):
        distance = convolution.measure_similarities(series, samples, False, num_processes = 10, filter_level = filter_level)
        clusters = np.argmax(distance, axis = 0)
        print iteration, np.count_nonzero(clusters - prev_clusters)
        prev_clusters = clusters
        #np.matrix(distance) * sequences
        prob = np.exp(distance * 10) 
        prob = prob / np.sum(prob, axis = 0)
        import time
        start = time.time()
        for i in range(10):
            #foo = np.multiply(sequences, prob[i][:, np.newaxis])
            #samples[i] = np.mean(foo, axis = 0)
            samples[i] = np.average(foo, axis = 0, weights = prob[i])
        print time.time() - start
        
        for i in range(k):
            foo = np.multiply(sequences, prob[i][:, np.newaxis])
            samples[i] = np.mean(foo, axis = 0)
    clusters[:m] = np.NaN
    for i in range(k):
        samples[i] = samples[i] - np.mean(samples[i])
    return clusters, samples, distance


clusters_train, samples, distance_train = cluster_mixture(data.price.values[:int(n/2)], k, m, num_iter = 50, filter_level = 1)
distance_test = convolution.measure_similarities(data.price.values[int(n/2):], samples, False, num_processes = 10, filter_level = 1)


dimension = 20
performance = np.zeros(k)
for i in range(k):
    points = np.where(clusters_train==i)[0]
    performance[i] = np.mean(data.p1800.values[:int(n/2)][points])
performance_rank = np.argsort(performance)
best_samples = np.zeros((dimension, m))
best_distance_train = np.zeros((dimension, len(distance_train[0])))
for i in range(int(dimension/2)):
    best_samples[i] = samples[performance_rank[i]]
    best_samples[-i-1] = samples[performance_rank[-i-1]]
    best_distance_train[i] = distance_train[performance_rank[i]]
    best_distance_train[-i-1] = distance_train[performance_rank[-i-1]]
    
best_distance_test = convolution.measure_similarities(data.price.values[int(n/2):], best_samples, False, num_processes = 10, filter_level = 1)

#ensemble model
prob = np.exp(best_distance_train * 10) 
prob = prob / np.sum(prob, axis = 0)
prob = np.asmatrix(prob)
coeff = prob * np.asmatrix(data.p1800.values[:int(n/2)]).transpose() / np.sum(prob, axis = 1)

prob = np.exp(best_distance_test * 10) 
prob = prob / np.sum(prob, axis = 0)
prob = np.asmatrix(prob)
_Y = prob.transpose() * coeff
_Y = np.array(_Y)[:, 0]
Y = data.p1800.values[int(n/2):]
np.corrcoef(_Y, Y)

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100)
p1.bar(bin_edges[:-1], statistic, width = 0.01)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100)
p2.bar(bin_edges[:-1], statistic, width = 0.01)
plt.show()