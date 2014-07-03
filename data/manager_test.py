'''
Created on Apr 7, 2014

@author: zhk
'''
from manager import Manager
from btce import Btce
from bitfinex import Bitfinex
from btcchina import BTCChina
from huobi import Huobi

if __name__ == '__main__':
    manager = Manager(False)
#    manager.load_ticker(Bitfinex(manager), 'ltc_btc')
#    manager.load_ticker(Btce(manager),     'ltc_usd')
#    manager.load_ticker(Btce(manager),     'ltc_btc')
    manager.load_ticker(BTCChina(manager), 'ltc_cny')
#    manager.load_ticker(BTCChina(manager), 'ltc_btc')
#    manager.load_ticker(Huobi(manager),    'ltc_cny')
    manager.start()