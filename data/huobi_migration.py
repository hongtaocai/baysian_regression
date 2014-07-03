# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 07:19:30 2014

@author: zhk
"""

import MySQLdb as sql
import pandas as pd
import ast

exchange = 'huobi'
conn = sql.connect(host = '128.31.6.245', user = 'jingxiao', passwd = 'jingxiao', db = 'bitcoin', port = 3308)
depth = pd.io.sql.frame_query('''SELECT * from huobi_depth1''', con=conn)

def parse(string):
    parts = string[2:-2].split('),(')
    parts.reverse()
    value = float(parts[0].split(',')[0])
    return "[("+"),(".join(parts)+")]", value
    
import time
t = time.time()
n = len(depth)
ask = np.zeros(n)
bid = np.zeros(n)
asks = np.ndarray(n, dtype = 'object')
bids = np.ndarray(n, dtype = 'object')
for i in range(0, n):
    if i%10000==0:
        print i
    asks[i], ask[i] = parse(depth.bids[i])
    bids[i], bid[i] = parse(depth.asks[i])
print time.time() - t

depth.bids = bids
depth.asks = asks
depth.bid = bid
depth.ask = ask
depth.to_sql('huobi_depth', conn, flavor='mysql', if_exists='replace', index=False, index_label=None)