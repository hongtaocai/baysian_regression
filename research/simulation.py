# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 19:32:24 2014

@author: zhk
"""
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

def simulate(data, signal, t_in, t_out, plot = False, short_allowed = False, delay = 0):
    timestamp = data.t.values        
    bid = data.bid.values
    ask = data.ask.values
    trades = np.zeros(signal.size)      
    position = 0        
    for index in range(signal.size-delay):
        if position > 0 and signal[index] < -t_out:
            trades[index+delay] = -1
            position = position - 1
        elif position < 0 and signal[index] > t_out:
            trades[index+delay] = 1
            position = position + 1
        elif position == 0:
            if signal[index] > t_in:
                trades[index+delay] = 1    
                position = position + 1
            elif signal[index] < -t_in and short_allowed:
                trades[index+delay] = -1   
                position = position - 1
    if position > 0:
        trades[-1]  = trades[-1] - 1
        position = position - 1
    elif position < 0:
        trades[-1]  = trades[-1] + 1
        position = position + 1
    counter = np.count_nonzero(trades)
    enter_indices = np.where(trades!=0)[0][np.arange(0, counter, 2)]
    exit_indices = np.where(trades!=0)[0][np.arange(1, counter, 2)]
    long_indices = np.where(trades[enter_indices] == 1)[0]
    short_indices = np.where(trades[enter_indices] == -1)[0]
    profits = np.zeros(int(counter/2))
    profits[long_indices] = -ask[enter_indices[long_indices]] + bid[exit_indices[long_indices]]
    profits[short_indices] = bid[enter_indices[short_indices]] - ask[exit_indices[short_indices]]
    cum_profits = np.cumsum(profits)
    pnl = cum_profits[-1]
    holding_times = timestamp[exit_indices] - timestamp[enter_indices]
    if plot:
        date = np.array([dt.datetime.fromtimestamp(ts) for ts in timestamp])
        mid = data.price.values
        buy_indices = np.where(trades == 1)[0]
        sell_indices = np.where(trades == -1)[0]
        fig = plt.figure()
        plt.plot(date, mid)
        #plt.plot(date[sell_indices], bid[sell_indices], 'o', color = 'red')
        #plt.plot(date[buy_indices], ask[buy_indices], 'o',  color = 'green')
        plt.xlabel('time', fontsize=28)
        plt.ylabel('bitcoin price (blue)', fontsize=28)
        plt.tick_params(axis='x', labelsize=24)
        plt.tick_params(axis='y', labelsize=24)
        plt.ylim(2650, 7150)
        ax2 = plt.twinx()
        ax2.plot(date[exit_indices], cum_profits, color = 'black')
        ax2.set_ylabel('profit (black)', fontsize=28)
        ax2.tick_params(axis='y', labelsize=24)
        ax2.set_ylim(0, 4500)
        plt.show()
        fig = plt.figure()
        p1 = fig.add_subplot(211)
        p1.hist(profits, bins = 60, range = (-30, 30))
        p1.set_xlim(-30, 30)
        p2 = fig.add_subplot(212)
        p2.hist(holding_times, bins = 60, range=(0, 1800))        
        plt.show()   
    return pnl, int(counter/2), pnl/counter*2.0, np.mean(holding_times), trades

    
def characterize_strategy(test, _Y, Y, lo = 0, hi = 3, step = 0.05, short_allowed = False, plot = True, delay = 0, exit_quick = False):
    num_trades = []
    pnls = []
    if plot:
        fig = plt.figure()
        p11 = fig.add_subplot(211)
        p12 = p11.twinx()
        p21 = fig.add_subplot(212)
        p22 = p21.twinx()
    for i in np.arange(lo, hi, step):
        if exit_quick:
            (pnl, num_trade, avg_pnl, avg_time, trades) = simulate(test, _Y, i, 0, short_allowed = short_allowed, delay = delay)
        else:
            (pnl, num_trade, avg_pnl, avg_time, trades) = simulate(test, _Y, i, i, short_allowed = short_allowed, delay = delay)
        accuracy = 1.0 * sum((_Y>0) & (Y>0)) / (sum((_Y>0))+0.00001)
        num_trades.append(num_trade)
        pnls.append(avg_pnl)
        print i, accuracy, avg_pnl, avg_time, num_trade, pnl
        if plot:
            bar = p21.plot(i, avg_pnl, 'bo')
            bar = p22.plot(i, pnl, 'ko')        
            foo = p11.plot(i, avg_time, 'bo')
            foo = p12.plot(i, num_trade, 'ko')
    if plot:
        p = p21.set_xlabel('threshold', fontsize=28)
        p = p11.set_xlim(lo,hi)
        p = p21.set_xlim(lo,hi)
        p = p11.set_ylabel('holding time (blue)', fontsize=28)
        p = p12.set_ylabel('sample size (black)', fontsize=28)
        p = p21.set_ylabel('pnl (blue)', fontsize=28)
        p = p22.set_ylabel('total pnl (black)', fontsize=28)
        p = p22.set_ylim(0, 4500)
        p11.tick_params(axis='x', labelsize=24)
        p11.tick_params(axis='y', labelsize=24)
        p12.tick_params(axis='y', labelsize=24)
        p21.tick_params(axis='x', labelsize=24)
        p21.tick_params(axis='y', labelsize=24)
        p22.tick_params(axis='y', labelsize=24)
        p = plt.show()
        
        p = fig = plt.figure()
        p = plt.plot(pnls, num_trades, 'o')
        p = plt.tick_params(axis='x', labelsize=24)
        p = plt.tick_params(axis='y', labelsize=24)
        p = plt.xlabel('average profit', fontsize=28)
        p = plt.ylabel('number of trades', fontsize=28)
        p = plt.xlim(0,5)
        p = plt.ylim(0,10000)
        p = plt.show()