#!/usr/bin/env python
# -*- coding: utf-8 -*-

from exchange import Exchange
from time import time

class Btc796(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = '796'
        self.tick_query = '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, bid, ask, low, high, last, vol) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {})'''
        self.depth_interval = 10 
    
    def get_tick(self, ticker):
        if ticker == 'btc_usd':
            url = 'http://api.796.com/v3/spot/ticker.html?type=btcusd'
        elif ticker == 'ltc_usd':
            url = 'http://api.796.com/v3/spot/ticker.html?type=ltcusd'
        elif ticker == 'btc_usdw':
            url = 'http://api.796.com/v3/futures/ticker.html?type=weekly'
        elif ticker == 'ltc_usdw':
            url = 'http://api.796.com/v3/futures/ticker.html?type=ltc'
        requested = int(time()*1000000)
        response = self.get_data(url)
        data = response.json()['ticker']
        received = int(time()*1000000)
        bid = data['buy']
        ask = data['sell']
        low = data['low']
        high = data['high']
        last = data['last']
        vol = data['vol']
        query = self.tick_query.format(self.name, ticker, requested, received, bid, ask, low, high, last, vol)
        return data, float(bid), float(ask), query

    def get_depth(self, ticker = 'btc_cny'):
        if ticker == 'btc_usd':
            url = 'http://api.796.com/v3/spot/depth.html?type=btcusd'
        elif ticker == 'ltc_usd':
            url = 'http://api.796.com/v3/spot/depth.html?type=ltcusd'
        elif ticker == 'btc_usdw':
            url = 'http://api.796.com/v3/futures/depth.html?type=weekly'
        elif ticker == 'ltc_usdw':
            url = 'http://api.796.com/v3/futures/depth.html?type=ltc'        
        requested = int(time()*1000000)
        response = self.get_data(url)
        data = response.json()
        received = int(time()*1000000)
        data['asks'] = sorted([(float(d[0]), float(d[1])) for d in data['asks']])
        data['bids'] = sorted([(float(d[0]), float(d[1])) for d in data['bids']], reverse = True)
        (ask, ask_qty) = data['asks'][0]
        (bid, bid_qty) = data['bids'][0]
        micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
        bids = data['bids'].__str__().replace(' ','')
        asks = data['asks'].__str__().replace(' ','')
        query = self.depth_query.format(self.name + '_depth', ticker, requested, received, bid, ask, bids, asks)
        return data, micro, query
    
    def get_trade(self, ticker, since = None):
        if ticker == 'btc_usd':
            url = 'http://api.796.com/v3/spot/trades.html?type=btcusd'
        elif ticker == 'ltc_usd':
            url = 'http://api.796.com/v3/spot/trades.html?type=ltcusd'
        elif ticker == 'btc_usdw':
            url = 'http://api.796.com/v3/futures/trades.html?type=weekly'
        elif ticker == 'ltc_usdw':
            url = 'http://api.796.com/v3/futures/trades.html?type=ltc'
        response = self.get_data(url)
        data = response.json()
        if len(data) == 0:
            return None, None, None, None, None, None
        else:
            for trade in data:
                if trade['type'] == 'buy':
                    trade['type'] = 'bid'
                elif trade['type'] == 'sell':
                    trade['type'] = 'ask'
            last_trade = float(data[0]['price'])
            volume = float(data[0]['amount'])
            order_type = data[0]['type']
            query = '''INSERT IGNORE INTO 796_trade (ticker, id, timestamp, price, type, amount) VALUES ''' + ','.join(['''('{}', {}, {}, {}, '{}', {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], trade['date'], trade['price'], trade['type'], trade['amount'])])
            return data, last_trade, volume, order_type, None, query
        
if __name__ == '__main__':
    pass 