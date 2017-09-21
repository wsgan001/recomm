# -*- coding:utf-8 -*-
import time
from svd import *
#from loadsMovielensReviews import *

def relative_path(path):
	'''
	Return a path relative from fieldname
	'''
	dirname = os.path.abspath(os.path.join(os.path.dirname('__file__'),".."))
	#print os.path.realpath('__file__'),dirname
	#exit(0)
	path = os.path.join(dirname,path)
	#print path
	#print os.path.normpath(path)
	return os.path.normpath(path)

data = relative_path('data\ml-100k\u.data')
item = relative_path('data\ml-100k\u.item')
pick_path = relative_path('pickle\/reccod.pickle')

model = Recommender(data)

# #将对象完全保存到reccod1.pickle
# model.build('reccod1.pickle')

rec = Recommender.load(pick_path)
for item in rec.top_rated(234):
	print "%i:%0.3f"%item
