import MySQLdb
f = open('query_log.txt', 'r')
conn = MySQLdb.connect(host = '128.31.6.245', user = 'zhk', passwd = 'G0373485x', db = 'bitcoin', port = 3308)
previous = ''
i = 0
for line in f.readlines():
    i = i + 1
    try:
        if line.startswith('INSERT IGNORE') and previous != '':
            conn.cursor().execute(previous)
            previous = line
        else:
            previous = previous + line
    except:
        print i, 'previous: ' + previous
        #print 'current:' + line
        break
conn.cursor().execute(previous)
