# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 22:26:13 2014

@author: zhk
"""
import ast
import numpy as np
import MySQLdb as sql
import pandas as pd
import datetime as dt
    
def moving_average(data, k):
    ret = np.cumsum(data, dtype=float)
    ret[k:] = (ret[k:] - ret[:-k])/k
    ret[:k] = ret[:k] / np.arange(1,k+1)  
    return ret

def moving_square_average(data, k):
    return moving_average(np.square(data), k)
    
def moving_std(data, k):
    average = moving_average(data,k)
    square_average = moving_square_average(data,k)
    moving_var = square_average - np.square(average)
    return np.sqrt(moving_var)

def tuple_to_array(tuples):
    a, b = zip(*tuples)
    return np.array(a), np.array(b)
    
def load_data(start_time, end_time, exchange, ticker, step, load_depth = True):   
    conn = sql.connect(host = '128.31.6.245', user = '', passwd = '', db = 'bitcoin', port = 3308)    
    depth = pd.io.sql.frame_query('''SELECT received/1000000 as timestamp, ask, bid, asks, bids FROM {}
                                 WHERE ticker = '{}' AND requested > {} AND requested < {} ORDER BY received'''.format(exchange + '_depth', ticker, (start_time-3600)* 1e6, (end_time+3600)* 1e6),
                              con=conn)
    depth['price'] = (depth.ask + depth.bid)/2                            
    data = pd.DataFrame({'t': range(start_time, end_time, step)})
    data['depth_index'] = np.searchsorted(depth.timestamp.values, np.arange(start_time, end_time, step), side = 'right') - 1
    data['bid'] = depth.bid[data.depth_index].values
    data['ask'] = depth.ask[data.depth_index].values
    data['price'] = (data['ask'] + data['bid'])/2
    if load_depth:
        data['a_ps'] = None
        data['a_qs'] = None
        data['a_dpth'] = 0.0
        data['a_mean'] = 0.0
        data['a_mean_qty'] = 0.0
        data['b_ps'] = None
        data['b_qs'] = None
        data['b_dpth'] = 0.0
        data['b_mean'] = 0.0
        data['b_mean_qty'] = 0.0    
        for i in range(0,len(data)):
            index = data.depth_index.values[i]
            asks = ast.literal_eval(depth.asks[index])
            bids = ast.literal_eval(depth.bids[index])
            data.a_ps[i], data.a_qs[i] = tuple_to_array(asks)
            data.b_ps[i], data.b_qs[i] = tuple_to_array(bids)
            data['a_dpth'][i] = ( data.a_ps[i].values[-1] - data.a_ps[i].values[0]) / len(asks)
            data['b_dpth'][i] = (-data.b_ps[i].values[-1] + data.b_ps[i].values[0]) / len(bids)
            data['a_mean_qty'][i] = np.mean(data.a_qs[i])
            data['b_mean_qty'][i] = np.mean(data.b_qs[i])
            data['a_mean'][i] = np.mean(data.a_qs[i] * data.a_ps[i]) / data['a_mean_qty'][i]
            data['b_mean'][i] = np.mean(data.b_qs[i] * data.b_ps[i]) / data['b_mean_qty'][i]
    return data, depth
#
#import time
#start = time.time()
#for i in range(2000):
#    asks = ast.literal_eval(depth.asks[i])
#    bids = ast.literal_eval(depth.bids[i])
#    a, b = zip(*asks)
#    data.a_ps[i], data.a_qs[i] = tuple_to_array(asks)
#    data.b_ps[i], data.b_qs[i] = tuple_to_array(bids)
#    data['a_dpth'][i] = ( data.a_ps[i].values[-1] - data.a_ps[i].values[0]) / len(asks)
#    data['b_dpth'][i] = (-data.b_ps[i].values[-1] + data.b_ps[i].values[0]) / len(bids)
#    data['a_mean_qty'][i] = np.mean(data.a_qs[i])
#    data['b_mean_qty'][i] = np.mean(data.b_qs[i])
#    data['a_mean'][i] = np.mean(data.a_qs[i] * data.a_ps[i]) / data['a_mean_qty'][i]
#    data['b_mean'][i] = np.mean(data.b_qs[i] * data.b_ps[i]) / data['b_mean_qty'][i]
#print time.time() - start
#    
#        
