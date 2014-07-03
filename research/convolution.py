# -*- coding: utf-8 -*-
"""
Created on Wed May 14 23:48:03 2014

@author: zhk
"""

import ctypes
import numpy as np
from scipy.signal import argrelmax, argrelmin
import multiprocessing
import bitrade.research.util as util

def measure_similarity(series, pattern, _l2 = False):
    series = series - 3000
    pattern = pattern - np.mean(pattern)
    pattern = pattern[::-1]
    convolved = np.convolve(series, pattern, mode = 'full')
    convolved = convolved[0:len(series)]
    average = util.moving_average(series, len(pattern))
    square_average = util.moving_square_average(series, len(pattern))
    offset = sum(pattern) * average
    convolved = convolved - offset 
    scale = np.sqrt(square_average - np.square(average)) * len(pattern) * np.std(pattern)
    scale[scale==0] = 1    
    convolved = convolved / scale
    convolved[0:(len(pattern)-1)] = 0
    convolved = np.nan_to_num(convolved)
    if _l2:
        l2 = square_average  - np.square(average) + np.sum(np.square(pattern))/len(pattern) - 2 * convolved / len(pattern)
        l2[0:(len(pattern)-1)] = np.inf
        return convolved, l2
    else:
        return convolved

def _measure_similarity(data, patterns, indices, convolution, l2 = None, filter_level = 6):
    data = util.moving_average(data, filter_level)
    _convolution = np.zeros(len(data))
    for i in indices:
        if l2 == None:
            pattern = util.moving_average(patterns[i], filter_level)[0:len(patterns[i]):1]
            for j in range(filter_level):
                _convolution[j:len(data):filter_level] = measure_similarity(data[j:len(data):filter_level], pattern, False)
            convolution[i][:] = _convolution
            pass
        else:
            pass # to be implemented
            
def measure_similarities(data, patterns, _l2 = False, num_processes = 1, filter_level = 1):
    n = len(data)
    k = len(patterns)
    processes = []
    manager = multiprocessing.Manager()
    results = [multiprocessing.Array(ctypes.c_double, n) for i in range(k)] 
    indices = np.array_split(np.arange(k), num_processes)
    for i in range(num_processes):
        if len(indices) > 0:
            p = multiprocessing.Process(target = _measure_similarity, args = (data, patterns, indices[i], results, None, filter_level))
            processes.append(p)
            p.start()
    for p in processes:   
        p.join()    
    distance = np.zeros((k,n))
    for i in range(k):
        distance[i] = np.frombuffer(results[i].get_obj())
    return distance

def convolution_metric(convolved, response, sample_size):
    convolved = np.convolve(convolved, np.ones(10)/10, mode='same')
    peaks = argrelmax(convolved)[0]
    peak_values = convolved[peaks]
    samples = np.argsort(peak_values)[::-1][:sample_size]
    response = response[peaks[samples]]
    weights = convolved[peaks[samples]]
    weights = weights / np.sum(weights)
    return np.sum(response * weights), np.std(response)
    
def l2_metric(l2, response, sample_size):
    l2 = np.convolve(l2, np.ones(10)/10, mode='same')
    peaks = argrelmin(l2)[0]
    peak_values = l2[peaks]
    samples = np.argsort(peak_values)[:sample_size]
    response = response[peaks[samples]]
    weights = l2[peaks[samples]]
    weights = np.exp(-weights + min(weights))
    weights = weights / np.sum(weights)
    return np.sum(response * weights), np.std(response)
    
    
import matplotlib.pyplot as plt

if __name__ == "__main__":    
    data = np.cumsum(np.random.rand(10000)-0.5)
    pattern = data[0:100]
    convolved, l2 = measure_similarity(data, pattern)
    
    
    convolved = np.convolve(convolved, np.ones(10)/10, mode='same')
    peaks = argrelmax(convolved)[0]
    peak_values = convolved[peaks]
    k = 5
    samples = np.argsort(peak_values)[::-1][:k*k]
    samples = sort(peaks[samples])
    plt.figure()
    for i in range(k*k):
        p = plt.subplot(k, k, i+1)
        index = samples[i]
        p.plot(data[(index - 99):(index+1)])
        #plt.title((index, convolved[index], l2[index]))
        plt.title(index)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    
    peaks = argrelmin(l2)[0]
    peak_values = l2[peaks]
    k = 5
    samples = np.argsort(peak_values)[:k*k]
    samples = sort(peaks[samples])
    plt.figure()
    for i in range(k*k):
        p = plt.subplot(k, k, i+1)
        index = samples[i]
        p.plot(data[(index - 99):(index+1)])
        #plt.title((index, convolved[index], l2[index]))
        plt.title(index)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    
    plt.figure()
    plt.plot(pattern)
    
    fig = plt.figure()
    p1 = fig.add_subplot(311)
    p2 = fig.add_subplot(312)
    p3 = fig.add_subplot(313)
    p1.plot(data)
    p2.plot(convolved)
    p3.plot(-l2)
    np.corrcoef(convolved[100:], l2[100:])