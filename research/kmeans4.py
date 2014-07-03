# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 23:39:07 2014

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
    
def kmeans_fit(price_train, p1800_train, k, m, dimension):
    clusters_train, samples, distance_train = cluster(price_train, k, m, num_iter = 50, filter_level = 1)    
    performance = np.zeros(k)
    for i in range(k):
        points = np.where(clusters_train==i)[0]
        performance[i] = np.mean(p1800_train[points])
    performance_rank = np.argsort(performance)
    best_samples = np.zeros((dimension, m))
    best_distance_train = np.zeros((dimension, len(distance_train[0])))
    for i in range(int(dimension/2)):
        best_samples[i] = samples[performance_rank[i]]
        best_samples[-i-1] = samples[performance_rank[-i-1]]
        best_distance_train[i] = distance_train[performance_rank[i]]
        best_distance_train[-i-1] = distance_train[performance_rank[-i-1]]        
    prob = np.exp(best_distance_train * 10) 
    prob = prob / np.sum(prob, axis = 0)
    prob = np.asmatrix(prob)
    coeff = prob * np.asmatrix(p1800_train).transpose() / np.sum(prob, axis = 1)
    return best_samples, coeff
    
def kmeans_predict(price_test, best_samples, coeff):
    best_distance_test = convolution.measure_similarities(price_test, best_samples, False, num_processes = 10, filter_level = 1)        
    prob = np.exp(best_distance_test * 10)
    prob = prob / np.sum(prob, axis = 0)
    prob = np.asmatrix(prob)
    _Y = prob.transpose() * coeff    
    _Y = np.array(_Y)[:, 0]
    return _Y
    

exchange = 'okcoin'
ticker   = 'btc_cny'
start_time = 1392508800
end_time   = 1402444800
step = 10

if __name__ == "__main__":
    #data, depth = util.load_data(start_time, end_time, exchange, ticker, step, True)
    data = pickle.load(open( "data.p", "rb" ))
    forward  = 300
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p300'] = -depth.price[data.depth_index].values + depth.price[indices].values
    forward  = 600
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p600'] = -depth.price[data.depth_index].values + depth.price[indices].values
    forward  = 900
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p900'] = -depth.price[data.depth_index].values + depth.price[indices].values
    forward  = 1200
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p1200'] = -depth.price[data.depth_index].values + depth.price[indices].values
    forward  = 1800
    indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
    data['p1800'] = -depth.price[data.depth_index].values + depth.price[indices].values
  
    k = 100
    dimension = 20
    
    m = 45
    best_samples, coeff = kmeans_fit(data.price.values[:654000], data.p1800.values[:654000], k, m, dimension)
    data['s45'] = kmeans_predict(data.price.values, best_samples, coeff)
    data.s45=np.nan_to_num(data.s45)
    
    m = 90
    best_samples, coeff = kmeans_fit(data.price.values[:654000], data.p1800.values[:654000], k, m, dimension)
    data['s90'] = kmeans_predict(data.price.values, best_samples, coeff)
    
    m = 180
    best_samples, coeff = kmeans_fit(data.price.values[:654000], data.p1800.values[:654000], k, m, dimension)
    data['s180'] = kmeans_predict(data.price.values, best_samples, coeff)
    
    m = 360
    best_samples, coeff = kmeans_fit(data.price.values[:654000], data.p1800.values[:654000], k, m, dimension)
    data['s360'] = kmeans_predict(data.price.values, best_samples, coeff)
    
    m = 720
    best_samples, coeff = kmeans_fit(data.price.values[:654000], data.p1800.values[:654000], k, m, dimension)
    data['s720'] = kmeans_predict(data.price.values, best_samples, coeff)
    
    m = 1440
    best_samples, coeff = kmeans_fit(data.price.values[:654000], data.p1800.values[:654000], k, m, dimension)
    data['s1440'] = kmeans_predict(data.price.values, best_samples, coeff)
    
    
    train = data[:654000]
    test = data[681000:]
    Y  = test.p1800.values
    
    for feature in ['s45', 's90', 's180', 's360', 's720', 's1440']:
        _Y = test[feature].values
        print feature, np.corrcoef(_Y, Y)[0,1]
    
    
    response = 'p1800'
    features = ['s45', 's90', 's180', 's360', 's720', 's1440']
    
    regr = linear_model.LinearRegression()
    regr.fit(train[features], train[response].values)
    X = test[features]
    Y = test[response].values
    _Y = regr.predict(X)
    print np.corrcoef(_Y, Y)
    
    
    fig = plt.figure()
    p1 = fig.add_subplot(211)
    statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100)
    p1.bar(bin_edges[:-1], statistic, width = 0.05)
    p2 = fig.add_subplot(212)
    statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100)
    p2.bar(bin_edges[:-1], statistic, width = 0.05)
    plt.show()