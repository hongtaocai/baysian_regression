#!/usr/bin/env python
# -*- coding: utf-8 -*-

from exchange import Exchange
from time import time

class BTCChina(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.trade_interval = 10
        self.depth_interval = 10
        self.name  = 'btcchina'
        self.tick_url  = 'https://data.btcchina.com/data/ticker?market={}'
        self.depth_url = 'https://data.btcchina.com/data/orderbook?market={}'
        self.trade_url = 'http://data.btcchina.com/data/historydata?market={}'
        self.tick_query = '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, bid, ask, low, high, last, vol) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {})'''
    
    def get_tick(self, ticker = 'btc_cny'):
        if ticker == 'btc_cny':
            _ticker = 'btccny'
        elif ticker == 'ltc_cny':
            _ticker = 'ltccny'
        elif ticker == 'ltc_btc':
            _ticker = 'ltcbtc'
        requested = int(time()*1000000)
        response = self.get_data(self.tick_url.format(_ticker))
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
        if ticker == 'btc_cny':
            _ticker = 'btccny'
        elif ticker == 'ltc_cny':
            _ticker = 'ltccny'
        elif ticker == 'ltc_btc':
            _ticker = 'ltcbtc'
        requested = int(time()*1000000)
        response = self.get_data(self.depth_url.format(_ticker))
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
    
    def get_first_trade(self, exchange, ticker, get_conn):
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(self.last_trade_query.format(self.name + '_trade', ticker))
            since = cur.fetchone()[0]
            conn.close()
        except:
            return None
        if since == None:
            since = 1
        return since
    
    def get_trade(self, ticker, since = None):
        if ticker == 'btc_cny':
            _ticker = 'btccny'
        elif ticker == 'ltc_cny':
            _ticker = 'ltccny'
        elif ticker == 'ltc_btc':
            _ticker = 'ltcbtc'
        if since == None:
            response = self.get_data(self.trade_url.format(_ticker))
        else:
            response = self.get_data(self.trade_url.format(_ticker) + '&since=' + str(since))
        data = response.json()
        if len(data) == 0:
            return None, None, None, None, since, None
        else:
            for trade in data:
                if trade['type'] == 'buy':
                    trade['type'] = 'bid'
                elif trade['type'] == 'sell':
                    trade['type'] = 'ask'
            since = int(data[-1]['tid'])
            last_trade = float(data[-1]['price'])
            volume = float(data[-1]['amount'])
            order_type = data[-1]['type']
            query = '''INSERT IGNORE INTO btcchina_trade (ticker, id, timestamp, price, type, amount) VALUES ''' + ','.join(['''('{}', {}, {}, {}, '{}', {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], trade['date'], trade['price'], trade['type'], trade['amount'])])
            return data, last_trade, volume, order_type, since, query
        
if __name__ == '__main__':
    pass
#    from time import sleep, strftime, localtime
#     btcchina = BTCChina()
#     since = 3786142
#     while True:
#         trades = btcchina.get_trades(since = since)
#         if len(trades) != 0:
#             since = trades[-1]['tid']
#         for trade in trades:
#             print trade['tid'], strftime('%H:%M:%S', localtime(int(trade['date']))), strftime('%H:%M:%S')
#         print
#    data =  btcchina.get_depth();
#    print len(data['bids']), len(data['asks'])  