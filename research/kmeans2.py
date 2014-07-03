# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 20:54:36 2014

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

data, depth = util.load_data(start_time, end_time, exchange, ticker, step, True)
forward  = 1800
indices = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step) + forward)
data['p1800'] = -depth.price[data.depth_index].values + depth.price[indices].values

m = 360
k = 100
n = len(data)


def cluster( series, k, m, num_iter = 50, filter_level = 1):
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

#using all 100 clusters    
clusters_train, samples, distance_train = cluster(data.price.values[:654000], k, m, num_iter = 50, filter_level = 1)
distance_test = convolution.measure_similarities(data.price.values[681000:], samples, False, num_processes = 10, filter_level = 1)

#ensemble model
prob = np.exp(distance_train * 10) 
prob = prob / np.sum(prob, axis = 0)
prob = np.asmatrix(prob)
coeff = prob * np.asmatrix(data.p1800.values[:654000]).transpose() / np.sum(prob, axis = 1)

prob = np.exp(distance_test * 10) 
prob = prob / np.sum(prob, axis = 0)
prob = np.asmatrix(prob)
_Y = prob.transpose() * coeff
_Y = np.array(_Y)[:, 0]
Y = data.p1800.values[681000:]
np.corrcoef(_Y, Y)

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100)
p1.bar(bin_edges[:-1], statistic, width = 0.01)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100)
p2.bar(bin_edges[:-1], statistic, width = 0.01)
plt.show()

#linear regression
regr = linear_model.LinearRegression()        
regr.fit(distance_train.transpose(), data.p1800.values[:654000])
_Y = regr.predict(distance_test.transpose())
Y = data.p1800.values[681000:]
np.corrcoef(_Y, Y)

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100)
p1.bar(bin_edges[:-1], statistic, width = 0.5)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100)
p2.bar(bin_edges[:-1], statistic, width = 0.5)
plt.show()




#using best out of 100 clusters    
clusters_train, samples, distance_train = cluster(data.price.values[:654000], k, m, num_iter = 30, filter_level = 1)
dimension = 20
performance = np.zeros(k)
for i in range(k):
    points = np.where(clusters_train==i)[0]
    performance[i] = np.mean(data.p1800.values[:654000][points])
performance_rank = np.argsort(performance)
best_samples = np.zeros((dimension, m))
best_distance_train = np.zeros((dimension, len(distance_train[0])))
for i in range(int(dimension/2)):
    best_samples[i] = samples[performance_rank[i]]
    best_samples[-i-1] = samples[performance_rank[-i-1]]
    best_distance_train[i] = distance_train[performance_rank[i]]
    best_distance_train[-i-1] = distance_train[performance_rank[-i-1]]
best_distance_test = convolution.measure_similarities(data.price.values[681000:], best_samples, False, num_processes = 10, filter_level = 1)

#ensemble model
prob = np.exp(best_distance_train * 11) 
prob = prob / np.sum(prob, axis = 0)
prob = np.asmatrix(prob)
coeff = prob * np.asmatrix(data.p1800.values[:654000]).transpose() / np.sum(prob, axis = 1)

prob = np.exp(best_distance_test * 15) 
prob = prob / np.sum(prob, axis = 0)
prob = np.asmatrix(prob)
_Y = prob.transpose() * coeff
_Y = np.array(_Y)[:, 0]
Y = data.p1800.values[681000:]
np.corrcoef(_Y, Y)

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100)
p1.bar(bin_edges[:-1], statistic, width = 0.01)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100)
p2.bar(bin_edges[:-1], statistic, width = 0.01)
plt.show()

fig = plt.figure()
p11 = fig.add_subplot(211)
p12 = p11.twinx()
p21 = fig.add_subplot(212)
p22 = p21.twinx()
num_trades = []
pnls = []
for i in np.arange(0, 1, 0.02):
    signal = np.zeros(len(_Y))
    signal[_Y>i] = 1
    signal[_Y<-i] = -1
    (pnl, num_trade, avg_pnl, trades) = simulation.simulate(data.loc[681000:], signal, False, 1800)
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
p11.set_xlim(0,1)
p21.set_xlim(0,1)
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

#linear regression
regr = linear_model.LinearRegression()        
regr.fit(best_distance_train.transpose(), data.p1800.values[:654000])
_Y = regr.predict(best_distance_test.transpose())
Y = data.p1800.values[681000:]
np.corrcoef(_Y, Y)

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 200)
p1.bar(bin_edges[:-1], statistic, width = 0.05)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 200)
p2.bar(bin_edges[:-1], statistic, width = 0.05)
plt.show()


fig = plt.figure()
p11 = fig.add_subplot(211)
p12 = p11.twinx()
p21 = fig.add_subplot(212)
p22 = p21.twinx()
num_trades = []
pnls = []
for i in np.arange(1, 5, 0.1):
    signal = np.zeros(len(_Y))
    signal[_Y>i] = 1
    signal[_Y<-i] = -1
    (pnl, num_trade, avg_pnl, trades) = simulation.simulate(data.loc[681000:], signal, False, 1800)
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
p11.set_xlim(1,5)
p21.set_xlim(1,5)
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