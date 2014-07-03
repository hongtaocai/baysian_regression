'''
Created on Dec 18, 2013

@author: zhk
'''

from exchange import Exchange
from time import time

class Bitstamp(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = 'bitstamp'
        self.tick_url  = 'https://www.bitstamp.net/api/ticker/' 
        self.depth_url = 'https://www.bitstamp.net/api/order_book/'
        self.trade_url = 'http://www.bitstamp.net/api/transactions?time=minute' 
        self.tick_interval  = 2
        self.depth_interval = 20
        
    def get_tick(self, ticker = 'btc_usd'):
        requested = int(time()*1000000)
        response = self.get_data(self.tick_url)
        data = response.json()
        received = int(time()*1000000)
        timestamp =  int(float(data['timestamp']) * 1000000)
        bid = data['bid']
        ask = data['ask']
        low = data['low']
        high = data['high']
        last = data['last']
        vol = data['volume']
        query = self.tick_query.format(self.name, ticker, requested, received, timestamp, bid, ask, low, high, last, vol)
        return data, float(bid), float(ask), query
    
    def get_depth(self, ticker = 'btc_usd'):
        requested = int(time()*1000000)
        response = self.get_data(self.depth_url)
        data = response.json()
        received = int(time()*1000000)
        data['bids'] = sorted([(float(row[0]), float(row[1])) for row in data['bids']], reverse = True)
        data['asks'] = sorted([(float(row[0]), float(row[1])) for row in data['asks']])
        (ask, ask_qty) = data['asks'][0]
        (bid, bid_qty) = data['bids'][0]
        micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
        bids = data['bids'].__str__().replace(' ','')
        asks = data['asks'].__str__().replace(' ','')
        query = self.depth_query.format(self.name + '_depth', ticker, requested, received, bid, ask, bids, asks)
        return data, micro, query
    
    def get_trade(self, ticker = 'btc_usd', since = None):
        response = self.get_data(self.trade_url)
        data = response.json()
        if len(data) == 0:
            return None, None, None, None, None, None
        else:
            last_trade = float(data[0]['price'])
            volume = float(data[0]['amount'])
            query = '''INSERT IGNORE INTO bitstamp_trade (ticker, id, timestamp, price, amount) VALUES ''' + ','.join(['''('{}', {}, {}, {}, {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], trade['date'], trade['price'], trade['amount'])])
            return data, last_trade, volume, None, None, query
        
if __name__ == '__main__':
    #from time import strftime, localtime, sleep
    bitstamp = Bitstamp()
    while True:
        data =  bitstamp.get_trade()
        print data
        break
        #print strftime('%H:%M:%S', localtime(int(data['now'])/1000000)), strftime('%H:%M:%S', localtime(int(data['cached'])/1000000)), strftime('%H:%M:%S'), len(data['asks']), len(data['bids'])
    #    sleep(1)