# -*- coding:utf-8 -*-
import csv
from datetime import *
import os
from collections import defaultdict
import heapq
from operator import itemgetter
from math import sqrt
import numpy as np
import pickle
import time

def load_reviews(path,**kwargs):
	'''
	Loads Movielens reviews 
	u.data 196	242	3	881250949
	'''
	options = {
	'fieldnames':('userid','movieid','rating','timestamp'),
	'delimiter':'\t'
	}
	options.update(kwargs)

	parse_date = lambda r,k:datetime.fromtimestamp(float(r[k]))
	parse_int = lambda r,k: int(r[k])

	with open(path,'rb') as reviews:
		reader = csv.DictReader(reviews,**options)
		for row in reader:
			# print row 
			# break
			# {'movieid': '242', 'userid': '196', 'timestamp': '881250949', 'rating': '3'}
			row['movieid'] = parse_int(row,'movieid')
			row['userid'] = parse_int(row,'userid')
			row['rating'] = parse_int(row,'rating')
			row['timestamp'] = parse_date(row,'timestamp')
			# print row 
			# {'movieid': 196, 'userid': 196, \
			# 'timestamp': datetime.datetime(1997, 12, 4, 23, 55, 49), 'rating': 3}
			# break
			yield row 

def initialize(R,K):
	'''
	Returns initial matrices for an N X M matrix
	R:The matrix to be factorized,要分解的矩阵
	K:隐变量的个数
	返回值：P Q 分解结果
	'''
	N,M = R.shape
	P = np.random.rand(N,K)
	Q = np.random.rand(M,K)
	return P,Q

def factor(R,P=None,Q=None,K=2,steps=5000,alpha=0.0002,beta=0.02):
	'''
	利用给定参数对矩阵R进行分解
	R：给定矩阵
	P,Q：初始化矩阵 N*K M*K
	K：隐变量个数
	steps:优化迭代次数
	alpha:梯度下降学习速率
	beta:正则化参数
	'''
	if not P or not Q:
		P,Q = initialize(R,K)
	Q = Q.T
	rows,cols = R.shape
	for step in xrange(steps):
		for i in xrange(rows):
			for j in xrange(cols):
				if R[i,j]>0:
					eij = R[i,j] - np.dot(P[i,:],Q[:,j])
					for k in xrange(K):
						P[i,k] = P[i,k] + alpha*(2*eij*Q[k,j] - beta*P[i,k])
						Q[k,j] = Q[k,j] + alpha*(2*eij*P[i,k] - beta*Q[k,j])

		e = 0
		for i in xrange(rows):
			for j in xrange(cols):
				if R[i,j]>0:
					e = e + pow(R[i,j] - np.dot(P[i,:],Q[:,j]),2)
					for k in xrange(K):
						e = e +(beta/2)*(pow(P[i,k],2)+pow(Q[k,j],2))
		print step,e
		if e < 0.001:
			break
	return P,Q.T

class Recommender(object):
	"""docstring for Recommender"""
	def __init__(self,udata):
		self.udata = udata
		self.users = None
		self.movies = None
		self.reviews = None

		self.build_start = None
		self.build_finish = None
		self.description = None

		self.model = None
		self.features = 2
		self.steps = 5000
		self.alpha = 0.0002
		self.beta = 0.02

		self.load_dataset()

	def load_dataset(self):
		'''
		Loads the two datasets into momory ,indexed on the ID.
		'''
		self.users = set([])
		self.movies = set([])
		for review in load_reviews(self.udata):
			self.users.add(review['userid'])
			self.movies.add(review['movieid'])
		#print type(self.users)  <type 'set'>
		self.users = sorted(self.users)
		self.movies = sorted(self.movies)
		#print type(self.users)  <type 'list'>
		
		#构建矩阵
		self.reviews = np.zeros(shape=(len(self.users),len(self.movies)))
		for review in load_reviews(self.udata):
			uid = self.users.index(review['userid'])
			mid = self.movies.index(review['movieid'])
			self.reviews[uid,mid] = review['rating']

	@classmethod
	def load(klass,pickle_path):
		with open(pickle_path,'rb') as pkl:
			return pickle.load(pkl)
			
	def dump(self,pickle_path):
		with open(pickle_path,'wb') as pkl:
			return pickle.dump(self,pkl)

	def build(self,output=None):
		print output
		options = {
		'K':self.features,
		'steps':self.steps,
		'alpha':self.alpha,
		'beta':self.beta,
		}		
		self.build_start = time.time()
		self.P,self.Q = factor(self.reviews,**options)
		self.model = np.dot(self.P,self.Q.T)
		self.build_finish = time.time()
		#print output
		if output:
			self.dump(output)

	def predict_ranking(self,user,movie):
		uidx = self.users.index(user)
		midx = self.movies.index(movie)
		if self.reviews[uidx,midx]>0:
			return None
		#print self.model[uidx,midx]
		return self.model[uidx,midx]

	def top_rated(self,user,n=12):
		movies = [(mid,self.predict_ranking(user,mid))for mid in self.movies]
		return heapq.nlargest(n,movies,key=itemgetter(1))
