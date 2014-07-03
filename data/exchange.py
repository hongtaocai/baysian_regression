'''
Created on Feb 12, 2014

@author: zhk
'''


import requests
import threading
import traceback

from datetime import datetime
from time import sleep

class Exchange():
    lock = threading.Lock()
    query_log = open('query_log.txt', 'a')
    
    def __init__(self, manager):
        self.manager = manager
        self.tick_interval = 1
        self.trade_interval = 30
        self.depth_interval = 2
        self.error_interval = 60
        self.last_trade_query = '''SELECT MAX(id) FROM {} WHERE ticker = '{}' '''
        self.tick_query  = '''INSERT IGNORE INTO bitcoin_ticker (exchange, ticker, requested, received, timestamp, bid, ask, low, high, last, vol) VALUES ('{}', '{}', {}, {}, {}, {}, {}, {}, {}, {}, {})'''
        self.depth_query = '''INSERT IGNORE INTO {} (ticker, requested, received, bid, ask, bids, asks) VALUES ('{}', {}, {}, {}, {}, '{}', '{}')'''
        
    def get_data(self, url):
        return requests.get(url, timeout=10)
        
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
            since = 0
        return since
    
    def log_error(self):
        pass
    
    def process_tick(self, ticker, statuses, get_conn):
        while True:
            while self.manager.waiting:
                sleep(1)
            try:
                status = statuses[(self.name, ticker)]
                status['state'][0] = 'fetching'
                (data, bid, ask, query) = self.get_tick(ticker)
                status['state'][0] = 'sending'
                self.data_pipe(data)
                status['bid'] = bid
                status['ask'] = ask
                status['tick_update'] = datetime.now().strftime('%m-%d %H:%M:%S')
                status['state'][0] = 'saving'
                self.execute_query(query)
                status['state'][0] = 'sleeping'
                sleep(self.tick_interval)
            except:
                print self.name, ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                traceback.print_exc()
                sleep(self.error_interval)
    
    def process_depth(self, ticker, statuses, get_conn):
        while True:
            while self.manager.waiting:
                sleep(1)            
            try:
                status = statuses[(self.name, ticker)]
                status['state'][0] = 'fetching'
                (data, micro, query) = self.get_depth(ticker)
                status['state'][0] = 'sending'
                self.data_pipe(data)
                status['micro'] = micro
                status['depth_update'] = datetime.now().strftime('%m-%d %H:%M:%S')
                status['state'][0] = 'saving'
                self.execute_query(query)
                status['state'][0] = 'sleeping'
                sleep(self.depth_interval)
            except:
                print self.name, ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                traceback.print_exc()
                sleep(self.error_interval)

    def process_trade(self, ticker, statuses, get_conn):
        since = self.get_first_trade(self.name, ticker, get_conn)
        while True:
            while self.manager.waiting:
                sleep(1)
            try:
                status = statuses[(self.name, ticker)]
                status['state'][0] = 'fetching'
                (data, last_trade, volume, order_type, since, query) = self.get_trade(ticker, since)
                if data != None:
                    status['state'][0] = 'sending'
                    self.data_pipe(data)
                    status['last_trade'] = last_trade
                    status['volume'] = volume
                    status['type'] = order_type
                    status['trade_update'] = datetime.now().strftime('%m-%d %H:%M:%S')
                    status['state'][0] = 'saving'
                    self.execute_query(query)
                status['state'][0] = 'sleeping'
                sleep(self.depth_interval)
            except:
                print self.name, ticker, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                traceback.print_exc()
                sleep(self.error_interval)
    
    def execute_query(self, query):
        try:
            conn = self.manager.get_conn()
            conn.cursor().execute(query)
            conn.close()
        except:
            Exchange.lock.acquire()
            Exchange.query_log.write(query + '\n')
            Exchange.lock.release()

    def data_pipe(self, data):
        pass
        