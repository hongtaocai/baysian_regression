'''
Created on Dec 14, 2013

@author: zhk
'''
import json
import traceback

from datetime import datetime
from dateutil.parser import parse
from exchange import Exchange
from time import time, sleep

class Huobi(Exchange):
    def __init__(self, manager):
        Exchange.__init__(self, manager)
        self.name  = 'huobi'
        self.btc_url = 'http://market.huobi.com/staticmarket/detail.html'
        self.ltc_url = 'http://market.huobi.com/staticmarket/detail_ltc.html'
        self.tick_query = '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, bid, ask, low, high, last, vol) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {})'''

    def process_tick(self, ticker, statuses, get_conn):
        while True:
            while self.manager.waiting:
                sleep(1)
            try:
                status = statuses[(self.name, ticker)]
                status['state'][0] = 'fetching'
                requested = int(time()*1000000)
                if ticker == 'btc_cny':
                    response = self.get_data(self.btc_url)
                elif ticker == 'ltc_cny':
                    response = self.get_data(self.ltc_url)
                received = int(time()*1000000)
                if response.text.startswith('view_detail(') and response.text.endswith(')'):
                    data = json.loads(response.text[12:-1])
                    
                    bids = sorted([(float(d['price']), float(d['amount'])) for d in data['buys']], reverse = True)
                    asks = sorted([(float(d['price']), float(d['amount'])) for d in data['sells']])
                    (ask, ask_qty) = asks[0]
                    (bid, bid_qty) = bids[0]
                    micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
                    bids = bids.__str__().replace(' ','')
                    asks = asks.__str__().replace(' ','')
                    depth_query = self.depth_query.format(self.name + '_depth', ticker, requested, received, bid, ask, bids, asks)
                    
                    high = data['p_high']
                    low  = data['p_low']
                    last = data['p_last']
                    vol  = data['amount']
                    tick_query = self.tick_query.format(self.name, ticker, requested, received, bid, ask, low, high, last, vol)
                    
                    data['trades'] = data['trades'][0:2] 
                    trades = []
                    for trade in data['trades']:
                        timestamp = (parse(trade['time']) - datetime(1970,1,1)).total_seconds() + 16 * 3600
                        if timestamp > time():
                            timestamp = timestamp - 86400
                        elif  timestamp < time() - 86400:
                            timestamp = timestamp + 86400
                        price  = trade['price']
                        amount = trade['amount']
                        if trade['type'] == u'\u4e70\u5165':
                            trade['type'] = 'bid'
                        elif trade['type'] == u'\u5356\u51fa':
                            trade['type'] = 'ask'
                        trades = trades+ [ticker, timestamp, price, trade['type'], amount]
                    last_trade = float(data['trades'][0]['price'])
                    volume = float(data['trades'][0]['amount'])
                    order_type = data['trades'][0]['type']
                    trade_query = '''INSERT IGNORE INTO huobi_trade (ticker, timestamp, price, type, amount) VALUES ''' + ','.join(['''('{}', {}, {}, '{}', {}) '''] * (len(data['trades'])))
                    trade_query = trade_query.format(*trades)
                    
                    status['state'][0] = 'sending'
                    self.data_pipe(data)
                    status['bid'] = bid
                    status['ask'] = ask
                    status['tick_update'] = datetime.now().strftime('%m-%d %H:%M:%S')
                    status['micro'] = micro
                    status['depth_update'] = datetime.now().strftime('%m-%d %H:%M:%S')
                    status['last_trade'] = last_trade
                    status['volume'] = volume
                    status['type'] = order_type
                    status['trade_update'] = datetime.now().strftime('%m-%d %H:%M:%S')
                    status['state'][0] = 'saving'
                    self.execute_query(tick_query)
                    self.execute_query(depth_query)
                    self.execute_query(trade_query)
                    status['state'][0] = 'sleeping'
                    sleep(self.tick_interval)
            except:
                print self.name, ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                traceback.print_exc()
                sleep(self.error_interval)
        
    def process_depth(self, ticker, statuses, get_conn):
        pass
    
    def process_trade(self, ticker, statuses, get_conn):
        pass
    
#     def get_tick(self, ticker = 'btc_cny'):
#         requested = int(time()*1000000)
#         response = self.get_data(self.tick_url)
#         data = response.json()['ticker']
#         received = int(time()*1000000)
#         bid = data['buy']
#         ask = data['sell']
#         low = data['low']
#         high = data['high']
#         last = data['last']
#         vol = data['vol']
#         query = self.tick_query.format(self.name, ticker, requested, received, bid, ask, low, high, last, vol)
#         return data, float(bid), float(ask), query
#             
#     def get_depth(self, ticker = 'btc_cny'):
#         requested = int(time()*1000000)
#         response = self.get_data(self.depth_url)
#         data = response.json()
#         received = int(time()*1000000)
#         data['asks'] = sorted([(float(d[0]), float(d[1])) for d in data['asks']])
#         data['bids'] = sorted([(float(d[0]), float(d[1])) for d in data['bids']], reverse = True)
#         (ask, ask_qty) = data['asks'][0]
#         (bid, bid_qty) = data['bids'][0]
#         micro = (ask * bid_qty + bid * ask_qty ) / (bid_qty + ask_qty)
#         bids = data['bids'].__str__().replace(' ','')
#         asks = data['asks'].__str__().replace(' ','')
#         query = self.depth_query.format(self.name + '_depth', ticker, requested, received, bid, ask, bids, asks)
#         return data, micro, query
        
            
if __name__ == '__main__':
    huobi = Huobi()
    huobi.process_tick('btc_cny', None, None)