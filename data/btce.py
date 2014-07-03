'''
Created on Dec 18, 2013

@author: zhk
'''

from exchange import Exchange
from time import time

class Btce(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = 'btce'
        self.tick_url = 'https://btc-e.com/api/2/{}/ticker'
        self.depth_url = 'http://btc-e.com/api/2/{}/depth'
        self.trade_url = 'http://btc-e.com/api/2/{}/trades' 
        
    def get_tick(self, ticker = 'btc_usd'):
        requested = int(time()*1000000)
        response = self.get_data(self.tick_url.format(ticker))
        data = response.json()['ticker']
        received = int(time()*1000000)
        timestamp = int(data['updated']) * 1000000
        bid = data['buy']
        ask = data['sell']
        low = data['low']
        high = data['high']
        last = data['last']
        vol = data['vol_cur']
        query = self.tick_query.format(self.name, ticker, requested, received, timestamp, bid, ask, low, high, last, vol)
        return data, float(bid), float(ask), query
        
    def get_depth(self, ticker = 'btc_usd'):
        requested = int(time()*1000000)
        response = self.get_data(self.depth_url.format(ticker))
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
    
    def get_trade(self, ticker = 'btc_usd', since = None):
        response = self.get_data(self.trade_url.format(ticker))
        data = response.json()
        if len(data) == 0:
            return None, None, None, None, None, None
        else:
            last_trade = float(data[0]['price'])
            volume = float(data[0]['amount'])
            order_type = data[0]['trade_type']
            query = '''INSERT IGNORE INTO btce_trade (ticker, id, timestamp, price, type, amount) VALUES ''' + ','.join(['''('{}', {}, {}, {}, '{}', {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], trade['date'], trade['price'], trade['trade_type'], trade['amount'])])
            return data, last_trade, volume, order_type, None, query
        
if __name__ == '__main__':
    #from time import strftime, localtime, sleep
    btce = Btce()
    while True:
        data =  btce.get_trade()
        print data
        break
        #print strftime('%H:%M:%S', localtime(int(data['now'])/1000000)), strftime('%H:%M:%S', localtime(int(data['cached'])/1000000)), strftime('%H:%M:%S'), len(data['asks']), len(data['bids'])
    #    sleep(1)