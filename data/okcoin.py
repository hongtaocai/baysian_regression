'''
Created on Dec 18, 2013

@author: zhk
'''

from exchange import Exchange
from time import time

class Okcoin(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = 'okcoin'
        self.tick_url  = 'https://www.okcoin.com/api/ticker.do?symbol={}'
        self.depth_url = 'https://www.okcoin.com/api/depth.do?symbol={}'
        self.trade_url = 'https://www.okcoin.com/api/trades.do?symbol={}'
        self.tick_query = '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, bid, ask, low, high, last, vol) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {})'''
    
    def get_first_trade(self, exchange, ticker, get_conn):
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(self.last_trade_query.format(self.name + '_trade', ticker))
            since = cur.fetchone()[0] - 1
            conn.close()
        except:
            return None
        if since == None:
            since = 1
        return since
            
    def get_tick(self, ticker = 'btc_cny'):
        requested = int(time()*1000000)
        response = self.get_data(self.tick_url.format(ticker))
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
            
    def get_trade(self, ticker, since = None):
        if since == None:
            response = self.get_data(self.trade_url.format(ticker))
        else:
            response = self.get_data(self.trade_url.format(ticker) + '&since=' + str(since))
        data = response.json()
        if len(data) == 0:
            return None, None, None, None, since, None
        else:
            for trade in data:
                if trade['type'] == 'buy':
                    trade['type'] = 'bid'
                elif trade['type'] == 'sell':
                    trade['type'] = 'ask'            
            since = int(data[-1]['tid']) - 1
            last_trade = float(data[-1]['price'])
            volume = float(data[-1]['amount'])
            order_type = data[-1]['type']
            query = '''INSERT IGNORE INTO okcoin_trade (ticker, id, timestamp, price, type, amount) VALUES ''' + ','.join(['''('{}', {}, {}, {}, '{}', {}) '''] * (len(data)))
            query = query.format(*[item for trade in data for item in (ticker, trade['tid'], trade['date'], trade['price'], trade['type'], trade['amount'])])
            return data, last_trade, volume, order_type, since, query
        
if __name__ == '__main__':
    #from time import strftime, localtime, sleep
    okcoin = Okcoin()
    while True:
        data =  okcoin.get_trade(since = 0)
        print data
        #print strftime('%H:%M:%S', localtime(int(data['now'])/1000000)), strftime('%H:%M:%S', localtime(int(data['cached'])/1000000)), strftime('%H:%M:%S'), len(data['asks']), len(data['bids'])
    #    sleep(1)
        break