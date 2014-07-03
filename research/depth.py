# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:33:12 2014

@author: zhk
"""

from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import bitrade.research.simulation as simulation
import bitrade.research.util as util
import bitrade.research.simulation as simulation

exchange = 'okcoin'
ticker   = 'btc_cny'
start_time = 1392508800
end_time   = 1403568000
step = 10

data, depth = util.load_data(start_time, end_time, exchange, ticker, step, True)

delay = 0
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


indices = np.arange(0, 654000, 6)
d = data.loc[indices]
response = 'p1800'

fig = plt.figure()
p1 = fig.add_subplot(121, aspect='equal')
im1 = p1.hexbin(d.a_mean_qty, d.b_mean_qty, C=d[response], gridsize=50, cmap=None, bins=None, vmin = -50, vmax=50)
p1.axis([0, 15, 0, 15])
cb =plt.colorbar(im1)
p2 = fig.add_subplot(122, aspect='equal')
im2 = p2.hexbin(d.a_mean_qty, d.b_mean_qty, gridsize=50, cmap=None, bins=None, vmin = 0, vmax= 200)
p2.axis([0, 15, 0, 15])
cb =fig.colorbar(im2)
plt.show()

fig = plt.figure()
p1 = fig.add_subplot(121, aspect='equal')
im1 = p1.hexbin(d.a_mean - d.ask, d.bid - d.b_mean, C=d[response], gridsize=100, cmap=None, bins=None, vmin = -50, vmax=50)
p1.axis([0, 100, 0, 100])
cb =plt.colorbar(im1)
p2 = fig.add_subplot(122, aspect='equal')
im2 = p2.hexbin(d.a_mean - d.ask, d.bid - d.b_mean, gridsize=100, cmap=None, bins=None, vmin = 0, vmax= 200)
p2.axis([0, 100, 0, 100])
cb =fig.colorbar(im2)
plt.show()

fig = plt.figure()
p1 = fig.add_subplot(121, aspect='equal')
im1 = p1.hexbin(d.a_dpth, d.b_dpth, C=d[response], gridsize=100, cmap=None, bins=None, vmin = -50, vmax=50)
p1.axis([0, 3, 0, 3])
cb =plt.colorbar(im1)
p2 = fig.add_subplot(122, aspect='equal')
im2 = p2.hexbin(d.a_dpth, d.b_dpth, gridsize=100, cmap=None, bins=None, vmin = 0, vmax= 200)
p2.axis([0, 3, 0, 3])
cb =fig.colorbar(im2)
plt.show()


data['s1'] = (data.a_mean_qty - data.b_mean_qty) / (data.a_mean_qty + data.b_mean_qty)
data['s2'] = (data.a_mean - data.ask) -(data.bid - data.b_mean)
data['s3'] = data.a_dpth - data.b_dpth

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(d.s2.values, d[response], bins = 100, range = (-30, 30))
p1.bar(bin_edges[:-1], statistic, width = 0.2)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(d.s2.values, d[response], statistic = 'count', bins = 100, range = (-30, 30))
p2.bar(bin_edges[:-1], statistic, width = 0.2)
plt.show()

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(d.s3.values, d[response], bins = 100, range = (-.5, .5))
p1.bar(bin_edges[:-1], statistic, width = 0.01)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(d.s3.values, d[response], statistic = 'count', bins = 100, range = (-.5, .5))
p2.bar(bin_edges[:-1], statistic, width = 0.01)
plt.show()

response = 'p300'
features = ['s1', 's2', 's3']
train = data[:654000]
#train = train.loc[(abs(train.s1)>0.05) & (abs(train.s2)>3) & (abs(train.s3)>0.05)]
test = data[681000:]
#test = test.loc[(abs(test.s1)>0.05) & (abs(test.s2)>3) & (abs(test.s3)>0.05)]

regr = linear_model.LinearRegression()
regr.fit(train[features], train[response].values)
X = test[features]
Y = test[response].values
_Y = regr.predict(X)
print np.corrcoef(_Y, Y)[0,1]

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100, range = (-2.5, 2.5))
p1.bar(bin_edges[:-1], statistic, width = 0.02)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100, range = (-2.5, 2.5))
p2.bar(bin_edges[:-1], statistic, width = 0.02)
plt.show()

simulation.characterize_strategy(test, _Y, Y, 0, 1, 0.05, short_allowed = True, delay = 1)


for response in ['p300','p600','p900','p1200','p1800']:
    regr = linear_model.LinearRegression()
    regr.fit(train[features], train[response].values)
    X = test[features]
    Y = test[response].values
    _Y = regr.predict(X)
    print response, np.corrcoef(_Y, Y)[0,1]
    simulation.characterize_strategy(test, _Y, Y, 0, 1, 0.05, short_allowed = True, delay = 1)

def weighted_regression(train, response, step):
    response_abs = abs(response)
    new_train = train[response_abs!=0]
    new_response = response[response_abs!=0]
    for i in range(step, int(max(response)), step):
        new_train = np.concatenate((new_train, train[response_abs>i]))
        new_response = np.concatenate((new_response, response[response_abs>i]))      
    regr = linear_model.LinearRegression()
    regr.fit(new_train, new_response)
    return regr

regr = weighted_regression(train[features], train[response].values, 5)
X = test[features]
Y = test[response].values
_Y = regr.predict(X)
_Y = _Y - np.mean(_Y)
print np.corrcoef(_Y, Y)[0,1]

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100, range = (-5, 5))
p1.bar(bin_edges[:-1], statistic, width = 0.02)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100, range = (-5, 5))
p2.bar(bin_edges[:-1], statistic, width = 0.02)
plt.show()

simulation.characterize_strategy(test, _Y, Y, 1.5, 3, 0.05, short_allowed = True,delay = 1)
    
def weighted_log_regression(train, response, step):
    response_abs = abs(response)
    new_train = train[response_abs!=0]
    new_response = response[response_abs!=0]
    for i in range(step, int(max(response)), step):
        new_train = np.concatenate((new_train, train[response_abs>i]))
        new_response = np.concatenate((new_response, response[response_abs>i]))      
    regr = linear_model.LogisticRegression()
    regr.fit(new_train, np.sign(new_response))
    return regr    

regr = weighted_log_regression(train[features], train[response].values, 5)
X = test[features]
Y = test[response].values
_Y = regr.predict_proba(X)[:, 1]
_Y = _Y - np.mean(_Y)
print np.corrcoef(_Y, Y)

fig = plt.figure()
p1 = fig.add_subplot(211)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, bins = 100, range = (-0.1, 0.1))
p1.bar(bin_edges[:-1], statistic, width = 0.002)
p2 = fig.add_subplot(212)
statistic, bin_edges, binnumber = sp.stats.binned_statistic(_Y, Y, statistic = 'count', bins = 100, range = (-0.1, 0.1))
p2.bar(bin_edges[:-1], statistic, width = 0.002)
plt.show()

simulation.characterize_strategy(test, _Y, Y, 0, 0.05, 0.002, short_allowed = True, delay = 1)
