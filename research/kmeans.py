# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 18:20:36 2014

@author: zhk
"""

import numpy as np
import scipy as sp
from sklearn import linear_model
import bitrade.research.util as util
import bitrade.research.convolution as convolution
import bitrade.research.simulation as simulation

class Kmeans:
    def __init__(self):
        pass
    
    def cluster(self, series, k, m, num_iter = 50, filter_level = 1):
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
        self.series = series
        self.k = k
        self.m = m
        self.clusters = clusters
        self.samples = samples
        self.distance = distance
        return clusters, samples, distance        
        
    def fit_regression(self, responses, dimension):
        k = self.k        
        m = self.m
        clusters = self.clusters
        samples = self.samples
        distance = self.distance
        performance = np.zeros(k)
        for i in range(k):
            points = np.where(clusters==i)[0]
            performance[i] = np.mean(responses[points])
        performance_rank = np.argsort(performance)
        best_samples = np.zeros((dimension, m))
        best_distance = np.zeros((dimension, len(distance[0])))
        for i in range(int(dimension/2)):
            best_samples[i] = samples[performance_rank[i]]
            best_samples[-i-1] = samples[performance_rank[-i-1]]
            best_distance[i] = distance[performance_rank[i]]
            best_distance[-i-1] = distance[performance_rank[-i-1]]
        regr = linear_model.LinearRegression()        
        regr.fit(best_distance.transpose(), responses)
        self.regr = regr
        self.best_samples = best_samples
        self.best_distance = best_distance
        return regr
    
    def predict_regression(self, series):
        distance = convolution.measure_similarities(series, self.best_samples, False, num_processes = 10, filter_level = 1)
        return self.regr.predict(distance.transpose())

    def classify(self, series, samples = None):
        distance = convolution.measure_similarities(series, samples, False, num_processes = 10, filter_level = 1) 
        return np.argmax(distance, axis = 0)    
        
if __name__ == "__main__":
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
    
    kmeans_model = Kmeans()
    clusters_training, samples, distance = kmeans_model.cluster(data.price.values[:int(n/2)], k, m, num_iter = 50, filter_level = 1)
    regr = kmeans_model.fit_regression(data.p1800.values[:int(n/2)], 10)
    _Y = kmeans_model.predict_regression(data.price.values[int(n/2):])
    Y = data.p1800.values[int(n/2):]
    
    fig = plt.figure()
    p1 = fig.add_subplot(211)
    statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100)
    p1.bar(bin_edges[:-1], statistic, width = 0.05)
    p2 = fig.add_subplot(212)
    statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100)
    p2.bar(bin_edges[:-1], statistic, width = 0.05)
    plt.show()
    
    d = data
    training = d.iloc[0:int(len(d)/2)]
    test     = d.iloc[int(len(d)/2):]
    
    fig = plt.figure()
    p11 = fig.add_subplot(211)
    p12 = p11.twinx()
    p21 = fig.add_subplot(212)
    p22 = p21.twinx()
    num_trades = []
    pnls = []
    for i in np.arange(0, 2, 0.05):
        signal = np.zeros(len(_Y))
        signal[_Y>i] = 1
        signal[_Y<-i] = -1
        (pnl, num_trade, avg_pnl, trades) = simulation.simulate(test, signal, False, 1800)
        bar = p21.plot(i, avg_pnl, 'bo')
        bar = p22.plot(i, pnl, 'ko')
        
        accuracy = sum((signal==1) & (Y>0)) / sum((signal==1))
        foo = p11.plot(i, accuracy, 'bo')
        foo = p12.plot(i, num_trade, 'ko')
        
        num_trades.append(num_trade)
        pnls.append(avg_pnl)
        print i, accuracy, avg_pnl, num_trade, pnl
    
    p11.set_xlabel('threshold')
    p21.set_xlabel('threshold')
    p11.set_xlim(0,2)
    p21.set_xlim(0,2)
    p11.set_ylabel('accuracy (blue)')
    p12.set_ylabel('sample size (black)')
    p21.set_ylabel('pnl (blue)')
    p22.set_ylabel('total pnl (black)')
    plt.show()
    
    fig = plt.figure()
    plt.plot(pnls, num_trades, 'o')
    plt.xlabel('average profit')
    plt.ylabel('number of trades')
    plt.xlim(0,10)
    plt.show()