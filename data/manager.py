'''
Created on Feb 12, 2014

@author: zhk
'''
import MySQLdb
import threading

from datetime import datetime
from time import sleep

from btc796 import Btc796
from bitfinex import Bitfinex
from bitstamp import Bitstamp
from btcchina import BTCChina
from btce import Btce
from chbtc import Chbtc
from huobi import Huobi
from okcoin import Okcoin
    
class Manager():
    def __init__(self, write_to_db = True):
        self.statuses = {}
        self.exchanges = {}
        self.exchange_list = []
        self.waiting = False
        self.backup = False
        self.write_to_db = write_to_db
        
    def truncate_all_tables(self):
        conn = self.get_conn()
        c = conn.cursor()
        c.execute('show tables')
        for row in c:
            c.execute('truncate table ' + row[0])
        conn.close()
        
    def load_ticker(self, exchange, ticker):
        self.exchange_list.append((exchange.name, ticker))
        self.exchanges[(exchange.name, ticker)] = exchange
        self.statuses[(exchange.name, ticker)]  = {'exchange':exchange.name, 'ticker':ticker, 
                                                   'bid':None, 'ask':None, 'micro':None, 
                                                   'last_trade':None, 'volume':None, 'type':None,
                                                   'tick_update':None, 'trade_update':None, 'depth_update':None,
                                                   'state': [None, None, None]}
            
    def get_conn(self):
        if self.write_to_db:
            return MySQLdb.connect(host = '127.0.0.1', user = '', passwd = '', db = 'bitcoin', port = 3308)
        else:
            return None
    
    def test_conn(self, timeout = 1):
        try:
            conn = MySQLdb.connect(host = '128.31.6.245', user = '', passwd = '', db = 'bitcoin', port = 3308, connect_timeout = timeout)
            conn.close()
            return True
        except:
            return False
        
    def prettify(self, data, length = 9):
        if data == None:
            return ' ' * length
        elif type(data) is float:
            return ('{:' + str(length) + '.3f}').format(data)
        elif isinstance(data, basestring):
            data = data[0:length]
            if len(data) < length:
                data = ' ' * (length - len(data)) + data
            return data

    def start(self):
        for (key, exchange) in self.exchanges.iteritems():
            threading.Thread(target = exchange.process_tick,  args = (key[1], self.statuses, self.get_conn)).start()
            threading.Thread(target = exchange.process_depth, args = (key[1], self.statuses, self.get_conn)).start()
            threading.Thread(target = exchange.process_trade, args = (key[1], self.statuses, self.get_conn)).start()
        while True:
            print datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for (exchange, ticker) in self.exchange_list:
                status = self.statuses[(exchange, ticker)]
                print self.prettify(status['exchange']), self.prettify(status['ticker']), ' ', \
                      self.prettify(status['bid']), self.prettify(status['ask']), ' ', self.prettify(status['micro']), ' ', \
                      self.prettify(status['last_trade']), self.prettify(status['volume'], 7), self.prettify(status['type'], 4), ' ', \
                      status['tick_update'], ' ', status['depth_update'], ' ', status['trade_update']
            print
            if self.backup:
                if self.test_conn():
                    sleep(1)
                    self.waiting = True
                else:
                    self.waiting = False
            else:
                sleep(1)
            
if __name__ == '__main__':
    manager = Manager()
    manager.load_ticker(Btc796(manager),   'btc_usd')
    manager.load_ticker(Bitfinex(manager), 'btc_usd')
    manager.load_ticker(Bitstamp(manager), 'btc_usd')
    manager.load_ticker(Btce(manager),     'btc_usd')
    manager.load_ticker(Btc796(manager),   'btc_usdw')
      
    manager.load_ticker(BTCChina(manager), 'btc_cny')
    manager.load_ticker(Chbtc(manager),    'btc_cny')
    manager.load_ticker(Huobi(manager),    'btc_cny')
    manager.load_ticker(Okcoin(manager),   'btc_cny')

    manager.load_ticker(Btce(manager),     'ltc_usd')
    manager.load_ticker(Btc796(manager),   'ltc_usd')
    manager.load_ticker(Bitfinex(manager), 'ltc_usd')
    manager.load_ticker(Btce(manager),     'ltc_usd')
    manager.load_ticker(Btc796(manager),   'ltc_usdw')
    
    manager.load_ticker(BTCChina(manager), 'ltc_cny')     
    manager.load_ticker(Chbtc(manager),    'ltc_cny')
    manager.load_ticker(Huobi(manager),    'ltc_cny')
    manager.load_ticker(Okcoin(manager),   'ltc_cny')

    manager.load_ticker(Bitfinex(manager), 'ltc_btc')
    manager.load_ticker(BTCChina(manager), 'ltc_btc')    
    manager.start()
        
