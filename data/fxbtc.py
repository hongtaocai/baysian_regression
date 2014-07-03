#!/usr/bin/env python
# -*- coding: utf-8 -*-

from exchange import Exchange
from time import time

class Fxbtc(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = 'fxbtc'
        self.depth_interval = 15
        self.tick_interval  = 2
        self.tick_url  = 'https://data.fxbtc.com/api?op=query_ticker&symbol={}'
        self.depth_url = 'https://data.fxbtc.com/api?op=query_depth&symbol={}'
        self.trade_url = 'https://data.fxbtc.com/api?op=query_history_trades&symbol={}'
        self.tick_query = '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, bid, ask, low, high, last, vol) 
                          VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {})'''
    
    def get_tick(self, ticker = 'btc_cny'):
        requested = int(time()*1000000)
        response = self.get_data(self.tick_url.format(ticker))
        data = response.json()['ticker']
        received = int(time()*1000000)
        bid = data['bid']
        ask = data['ask']
        low = data['low']
        high = data['high']
        last = data['last_rate']
        vol = data['vol']
        query = self.tick_query.format(self.name, ticker, requested, received, bid, ask, low, high, last, vol)
        return data, float(bid), float(ask), query
    
    def get_depth(self, ticker = 'btc_cny'):
        requested = int(time()*1000000)
        response = self.get_data(self.depth_url.format(ticker))
        data = response.json()['depth']
        received = int(time()*1000000)
        data['asks'] = sorted([(float(d['rate']), float(d['vol']), int(d['count'])) for d in data['asks']])
        data['bids'] = sorted([(float(d['rate']), float(d['vol']), int(d['count'])) for d in data['bids']], reverse = True)
        (ask, ask_qty) = data['asks'][0][0:2]
        (bid, bid_qty) = data['bids'][0][0:2]
        micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
        bids = data['bids'].__str__().replace(' ','')
        asks = data['asks'].__str__().replace(' ','')
        query = self.depth_query.format(self.name + '_depth', ticker, requested, received, bid, ask, bids, asks)
        return data, micro, query
    
    def get_trade(self, ticker = 'btc_cny', since = None):
        if since == None:
            response = self.get_data(self.trade_url.format(ticker))
        else:
            response = self.get_data(self.trade_url.format(ticker) + '&since=' + str(since))
        data = response.json()['datas'] 
        if len(data) == 0:
            return None, None, None, None, since, None
        else:
            since = int(data[-1]['tid'])
            last_trade = float(data[-1]['rate'])
            volume = float(data[-1]['vol'])
            order_type = data[-1]['order']
            query = '''INSERT IGNORE INTO fxbtc_trade (ticker, id, timestamp, price, type, amount) VALUES ''' + ','.join(['''('{}', {}, {}, {}, '{}', {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], trade['date'], trade['rate'], trade['order'], trade['vol'])])
            return data, last_trade, volume, order_type, since, query
        
if __name__ == '__main__':
    fxbtc = Fxbtc()
    print fxbtc.get_trade()