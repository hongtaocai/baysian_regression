'''
Created on Dec 18, 2013

@author: zhk
'''

from exchange import Exchange
from time import time

class Mtgox(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = 'mtgox'
        self.tick_url    = 'http://data.mtgox.com/api/1/{}/ticker'
        self.depth_url   = 'http://data.mtgox.com/api/2/{}/money/depth/fetch'
        self.trade_url   = 'https://data.mtgox.com/api/2/{}/money/trades' 
        self.depth_query = '''INSERT IGNORE INTO {} (ticker, requested, received, timestamp, bid, ask, bids, asks) 
                           VALUES ('{}', {}, {}, {}, {}, {}, '{}', '{}')'''  
        self.depth_interval = 15
        
    def get_tick(self, ticker = 'btc_usd'):
        if ticker == 'btc_eur':
            _ticker = 'BTCEUR'
        elif ticker == 'btc_usd':
            _ticker = 'BTCUSD'
        elif ticker == 'btc_jpy':
            _ticker = 'BTCJPY'
        requested = int(time()*1000000)
        response = self.get_data(self.tick_url.format(_ticker))
        data = response.json()['return']
        received = int(time()*1000000)
        timestamp = int(data['now'])
        bid = data['buy']['value']
        ask = data['sell']['value']
        low = data['low']['value']
        high = data['high']['value']
        last = data['last']['value']
        vol = data['vol']['value']
        query = self.tick_query.format(self.name, ticker, requested, received, timestamp, bid, ask, low, high, last, vol)
        return data, float(bid), float(ask), query
    
    def get_depth(self, ticker):
        if ticker == 'btc_eur':
            _ticker = 'BTCEUR'
        elif ticker == 'btc_usd':
            _ticker = 'BTCUSD'
        elif ticker == 'btc_jpy':
            _ticker = 'BTCJPY'
        requested = int(time()*1000000)
        response = self.get_data(self.depth_url.format(_ticker))
        data = response.json()['data']
        received = int(time()*1000000)
        timestamp = int(data['cached'])
        data['asks'] = sorted([(float(d['price']), float(d['amount']), int(float(d['stamp']))) for d in data['asks']])
        data['bids'] = sorted([(float(d['price']), float(d['amount']), int(float(d['stamp']))) for d in data['bids']], reverse = True)
        (ask, ask_qty) = data['asks'][0][0:2]
        (bid, bid_qty) = data['bids'][0][0:2]
        micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
        bids = data['bids'].__str__().replace(' ','')
        asks = data['asks'].__str__().replace(' ','')
        query = self.depth_query.format(self.name + '_depth', ticker, requested, received, timestamp, bid, ask, bids, asks)
        return data, micro, query
    
    def get_trade(self, ticker, since = None):
        if ticker == 'btc_eur':
            _ticker = 'BTCEUR'
        elif ticker == 'btc_usd':
            _ticker = 'BTCUSD'
        elif ticker == 'btc_jpy':
            _ticker = 'BTCJPY'
        
        url = self.trade_url.format(_ticker)
        if since == None:
            response = self.get_data(url)
        else:
            response = self.get_data(url + '?since=' + str(since))
        data = response.json()['data']
        if len(data) == 0:
            return None, None, None, None, since, None
        else:
            since = int(data[-1]['tid'])
            last_trade = float(data[-1]['price'])
            volume = float(data[-1]['amount'])
            order_type = data[-1]['trade_type']
            query = '''INSERT IGNORE INTO mtgox_trade (ticker, id, timestamp, price, type, amount, properties, primary_) VALUES ''' + ','.join(['''('{}', {}, {}, {}, '{}', {}, '{}', '{}') '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], round(float(trade['tid'])/1000000.0), trade['price'],trade['trade_type'],trade['amount'],trade['properties'],trade['primary'])])
            return data, last_trade, volume, order_type, since, query
        
if __name__ == '__main__':
    #from time import strftime, localtime, sleep
    mtgox = Mtgox()
    while True:
        data =  mtgox.get_trade(currency = 'BTCEUR', since = 1387990000000000)
        print data
        #print strftime('%H:%M:%S', localtime(int(data['now'])/1000000)), strftime('%H:%M:%S', localtime(int(data['cached'])/1000000)), strftime('%H:%M:%S'), len(data['asks']), len(data['bids'])
    #    sleep(1)