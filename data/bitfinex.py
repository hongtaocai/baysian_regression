#!/usr/bin/env python
# -*- coding: utf-8 -*-

from exchange import Exchange
from time import time

class Bitfinex(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.depth_interval = 10
        self.name  = 'bitfinex'
        self.tick_url  = 'https://api.bitfinex.com/v1/ticker/{}'
        self.depth_url = 'https://api.bitfinex.com/v1/book/{}'
        self.trade_url = 'https://api.bitfinex.com/v1/trades/{}'
        self.tick_query= '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, timestamp, bid, ask, last) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {})'''
        self.last_trade_query = '''SELECT MAX(timestamp) FROM {} WHERE ticker = '{}' '''
        
    def get_tick(self, ticker = 'btc_usd'):
        requested = int(time()*1000000)
        if ticker == 'btc_usd':
            _ticker = 'btcusd'
        elif ticker == 'ltc_usd':
            _ticker = 'ltcusd'
        elif ticker == 'ltc_btc':
            _ticker = 'ltcbtc'
        response = self.get_data(self.tick_url.format(_ticker))
        data = response.json()
        received = int(time()*1000000)
        timestamp =  int(float(data['timestamp']) * 1000000)
        bid = data['bid']
        ask = data['ask']
        last = data['last_price']
        query = self.tick_query.format(self.name, ticker, requested, received, timestamp, bid, ask, last)
        return data, float(bid), float(ask), query
    
    def get_depth(self, ticker = 'btc_usd'):
        requested = int(time()*1000000)
        if ticker == 'btc_usd':
            _ticker = 'btcusd'
        elif ticker == 'ltc_usd':
            _ticker = 'ltcusd'
        elif ticker == 'ltc_btc':
            _ticker = 'ltcbtc'            
        response = self.get_data(self.depth_url.format(_ticker))
        data = response.json()
        received = int(time()*1000000)
        data['asks'] = sorted([(float(d['price']), float(d['amount']), int(float(d['timestamp']))) for d in data['asks']])
        data['bids'] = sorted([(float(d['price']), float(d['amount']), int(float(d['timestamp']))) for d in data['bids']], reverse = True)
        (ask, ask_qty) = data['asks'][0][0:2]
        (bid, bid_qty) = data['bids'][0][0:2]
        micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
        bids = data['bids'].__str__().replace(' ','')
        asks = data['asks'].__str__().replace(' ','')
        query = self.depth_query.format(self.name + '_depth', ticker, requested, received, bid, ask, bids, asks)
        return data, micro, query
    
    def get_trade(self, ticker = 'btcusd', since = None):
        if ticker == 'btc_usd':
            _ticker = 'btcusd'
        elif ticker == 'ltc_usd':
            _ticker = 'ltcusd'
        elif ticker == 'ltc_btc':
            _ticker = 'ltcbtc'   
        response = self.get_data(self.trade_url.format(_ticker))
        data = response.json()
        if len(data) == 0:
            return None, None, None, None, None, None
        else:
            last_trade = float(data[0]['price'])
            volume = float(data[0]['amount'])
            query = '''INSERT IGNORE INTO bitfinex_trade (ticker, timestamp, price, amount) VALUES ''' + ','.join(['''('btc_usd', {}, {}, {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (trade['timestamp'],trade['price'],trade['amount'])])
            return data, last_trade, volume, None, None, query
            
if __name__ == '__main__':
    bitfinex = Bitfinex()
    print bitfinex.get_depth()