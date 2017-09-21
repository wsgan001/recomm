'''
def fab(max): 
    n, a, b = 0, 0, 1 
    L = []
    while n < max: 
        print b 
        a, b = b, a + b 
        n = n + 1
    return L
'''
class Fab(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1

    def __iter__(self):
        return self

    def next(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()

def fun2():  
	print 'first' 
	yield 23
	print 'second'
	yield 5
#g1 = fun2()
#g1.next()
#g1.next()
#fun2().next()

# import csv
# reader = csv.DictReader(file('1.csv', 'rb'))
# for i in reader:
#     print i

# parse_int = lambda r,k:int(r[k])
# print parse_int

import datetime
import time 

# print time.time()
# print time.localtime()

# print dir(datetime)
# print (datetime.strptime("2007-03-04 21:08:12", "%Y-%m-%d %H:%M:%S"))

# L = []
# L = [x*x for x in range(1,11)]
# print(L)
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# a = {1:'abs', 514:'fsafd'}
# for x in a :
#     print x

# review ={1: {'movieid': 1, 'userid': 1, 'timestamp': datetime.datetime(1997, 9, 23, 6, 2, 38), 'rating': 5}, 
#  2: {'movieid': 2, 'userid': 1, 'timestamp': datetime.datetime(1997, 10, 15, 13, 26, 11), 'rating': 3}, 
#  3: {'movieid': 3, 'userid': 1, 'timestamp': datetime.datetime(1997, 11, 3, 15, 42, 40), 'rating': 4}}
# users = set([])
# users.add(review['userid'])
# print type(users)

# import numpy as np


# a = [[1,2,3],[4,5,6]]
# a = np.array(a)
# b = [[1,2],[4,5],[3,6]]
# b = np.array(b)
# for i in xrange(3):
#     print np.dot(a[i,:])

