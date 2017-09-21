# -*- coding:utf-8 -*-
import csv
from datetime import *
import os
from collections import defaultdict
import heapq
from operator import itemgetter
from math import sqrt
import numpy as np

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

def relative_path(path):
	'''
	Return a path relative from fieldname
	'''
	dirname = os.path.dirname(os.path.realpath('__file__'))
	path = os.path.join(dirname,path)
	#print os.path.normpath(path)
	return os.path.normpath(path)

def load_movies(path,**kwargs):
	'''
	Loads Moiveslens movies
	u.item  \
	1|Toy Story (1995)|01-Jan-1995||\
	http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)\
	|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
	'''
	options = {
	'fieldnames':('movieid','title','release','video','url'),\
	'delimiter':'|','restkey':'genre'}
	options.update(kwargs)
	#print type(options['fieldnames'])

	parse_int = lambda r,k:int(r[k])
	parse_date = lambda r,k:datetime.strptime(r[k],'%d-%b-%Y') if r[k] else None

	with open(path,'rb') as movies:
		reader = csv.DictReader(movies,**options)
		#print reader #movies 为CSV的实例化对象，options为CSV的各种参数
		for row in reader:
			#以下三行主要用于数据类型转换，csv用于登入数据，movie
			row['movieid'] = parse_int(row,'movieid')
			row['release'] = parse_date(row,'release')
			row['video'] = parse_date(row,'video')
			#print row
			# {'movieid': 1, 'title': 'Toy Story (1995)', 、
			# 'url': 'http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)',\
			#  'genre': ['0', '0', '0', '1', '1', '1', '0', '0', '0', '0', '0',\
			#   '0', '0', '0', '0', '0', '0', '0', '0'], \
			#   'video': None, 'release': datetime.datetime(1995, 1, 1, 0, 0)}
			#  ..........
			yield row

class Moiveslens(object):
	"""Data structure to build our recommendation model on"""
	def __init__(self,udata,uitem):
		self.udata = udata
		self.uitem = uitem
		self.movies = {}
		self.reviews = defaultdict(dict)
		# print self.reviews
		# defaultdict(<type 'dict'>, {})
		# exit()
		self.load_dataset()

	def load_dataset(self):
		'''
		Loads the two datasets into momory ,indexed on the ID.
		'''
		for movie in load_movies(self.uitem):
			self.movies[movie['movieid']] = movie
			#print self.movies {1: {'movieid': 1, 'title': 'Toy Story (1995)', ' 创建电影字典
			#break
			# print self.movies[movie['movieid']])
			# {'movieid': 1682,\
			#  'title': 'Scream of Stone (Schrei aus Stein) (1991)',\
			#  'url': 'http://us.imdb.com/M/title-exact?Schrei%20aus%20Stein%20(1991)', \
			#  'genre': ['0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0'\
			#   , '0', '0', '0', '0', '0', '0'],\
			#   'video': None, \
			#   'release': datetime.datetime(1996, 3, 8, 0, 0)}
		for review in load_reviews(self.udata):
			# print review
			# {'movieid': 196, 'userid': 242, \
			# 'timestamp': datetime.datetime(1997, 12, 4, 23, 55, 49), \
			# 'rating': 3}

			self.reviews[review['userid']][review['movieid']] = review
			#print self.reviews
			# break
			# defaultdict(<type 'dict'>,\
			#  {196(userid): {242(movieid): {'movieid': 242, 'userid': 196, \
			#  'timestamp': datetime.datetime(1997, 12, 4, 23, 55, 49), \
			#  'rating': 3}}})
			# ....

	def reviews_for_movie(self,movieid):
		'''
		Yield the review for a given movie
		'''
		for review in self.reviews.values():
			if movieid in review:
				yield review[movieid]

	def average_reviews(self):
		'''
		average the star rating for all movies
		'''
		#print self.movies.keys()
		for movieid in self.movies:
			reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
			#list 生成器：self.reviews_for_movie中的r['rating']不断生成列表
			#print movieid,reviews
			#movieid:1[5, 4, 4, 4, 4, 3, 1, 5, 4, 5, 3, 5, 5, 5, 3, 5, 4, 5, 5, 4, 5, 2]
			average = sum(reviews)/float(len(reviews))
			yield(movieid,average,len(reviews))

	def bayesian_average(self,c = 59,m = 3):
		#c=59 预设值的选择
		for movieid in self.movies:
			reviews = list(r['rating'] for r in self.reviews_for_movie(movieid))
			average = ((c*m)+sum(reviews))/float(c+len(reviews))
			yield(movieid,average,len(reviews))

	def top_rated(self,n=10):
		'''
		yields the n top rated movies
		'''
		#return heapq.nlargest(n,self.average_reviews(),key = itemgetter(1))
		return heapq.nlargest(n,self.bayesian_average(),key = itemgetter(1))

	def shared_perferneces(self,criticA,criticB):
		'''
		找出被A与B共同评分过的电影，返回共同的电影评分
		'''
		if criticA not in self.reviews:
			raise KeyError("Couldn`t find critic '%s' in data "%criticA)
		if criticB not in self.reviews:
			raise KeyError("Couldn`t find critic '%s' in data "%criticB)
		#print self.reviews[criticA].keys(),type(self.reviews[criticA].keys())
		#exit(0)
		moviesA = set(self.reviews[criticA].keys())
		moviesB = set(self.reviews[criticB].keys())
		#print moviesA,moviesB
		shared = moviesA & moviesB
		#print shared,175
		#exit(0)
		reviews = {}
		for movieid in shared:
			# print movieid,179
			# print self.reviews[criticA][movieid]['rating']
			# print self.reviews[criticB][movieid]['rating']
			# break
			reviews[movieid] = (
				self.reviews[criticA][movieid]['rating'],
				self.reviews[criticB][movieid]['rating'],
				)
		return reviews

	def euclidean_distance(self,criticA,criticB):
		'''
		利用共同电影偏好计算用户距离（欧氏距离）：评分之差的平方 再做和运算
		'''
		perferneces = self.shared_perferneces(criticA,criticB)
		# print perferneces
		# {1: (4, 5), 515: (2, 5), 4: (4, 5), 8: (2, 5)
		if len(perferneces) == 0:
			return 0
		sum_of_squares = sum([pow(a-b,2) for a,b in perferneces.values()])

		#曼哈顿距离
		#sum_of_squares = sum([abs[a-b] for a,b in perferneces.values()])
		return 1/(1+sqrt(sum_of_squares))

	def pearson_correlation(self,criticA,criticB):
		'''
		皮尔逊相关性
		'''
		perferneces = self.shared_perferneces(criticA,criticB)
		length = len(perferneces)
		if length == 0: return 0
		sumA = sumB = sumSquareA = sumSquareB = sumProducts = 0
		for a,b in perferneces.values():
			sumA += a
			sumB += b
			sumSquareA += pow(a,2)
			sumSquareB += pow(b,2)
			sumProducts += a*b

		numerator = (sumProducts*length) - (sumA*sumB)
		denominator = sqrt(((sumSquareA*length)-pow(sumA,2))*
			((sumSquareB*length)-pow(sumB,2)))
		if denominator == 0 : return 0
		return abs(numerator/denominator)

	def similar_critics(self,user,metric='euclidean',n=None):
		'''
		寻找最匹配用户，返回top-n最相似用户
		'''
		metrics = {
		'euclidean':self.euclidean_distance,
		'pearson':self.pearson_correlation,
		}
		distance = metrics.get(metric,None)
		#print distance
		#handle problem that might occur
		if user not in self.reviews:
			raise KeyError("Unknow user,'%s'."%user)
		if not distance or not callable(distance):
			raise KeyError("Unknow or unprogrammed distance metric '%s'."%metric)
		#Compute user to critic similarities for all critics
		critics = {}
		for critic in self.reviews:
			#Don`t compare against yourself
			if critic == user:
				continue
			critics[critic] = distance(user,critic) #user与critic的距离计算
		if n:
			return heapq.nlargest(n,critics.items(),key=itemgetter(1))
		return critics #33:1.000 返回指定用户的相似用户
	
	def predict_ranking(self,user,movie,metric='euclidean',critics=None):
		'''
		电影的评分预测：该电影的加权均值，权重为评过分的用户与当前用户的相似度
		'''
		critics = critics or self.similar_critics(user,metric=metric)
		total = 0.0
		simsum = 0.0

		for critic,similarity in critics.items():
			if movie in self.reviews[critic]:
				total += similarity*self.reviews[critic][movie]['rating']
				simsum += similarity
		if simsum == 0.0:return 0.0
		return total/simsum

	def predict_all_rankings(self,user,metric='euclidean',n=None):
		'''
		预测所有电影评分
		'''
		critics = self.similar_critics(user,metric=metric)
		movies = {
		movie:self.predict_ranking(user,movie,metric,critics)
		for movie in self.movies
		}
		if n:
			return heapq.nlargest(n,movies.items(),key=itemgetter(1))
		return movies

	def shared_critics(self,movieA,movieB):
		'''
		计算电影的相似性
		'''
		if movieA not in self.movies:
			raise KeyError("Could`t find movie '%s' in data"%movieA)
		if movieB not in self.movies:
			raise KeyError("Could`t find movie '%s' in data"%movieB)
		criticsA = set(critic for critic in self.reviews if movieA in self.reviews[critic])
		criticsB = set(critic for critic in self.reviews if movieB in self.reviews[critic])
		shared = criticsA&criticsB

		reviews={}
		for critic in shared:
			reviews[critic] = (
				self.reviews[critic][movieA]['rating'],
				self.reviews[critic][movieB]['rating']
				)
		return reviews

	def similar_items(self,movie,metric='euclidean',n=None):
		metrics = {
		'euclidean':self.euclidean_distance,
		'pearson':self.pearson_correlation,
		}
		distance = metrics.get(metric,None)

		if movie not in self.reviews:
			raise KeyError("Unknow movie，'%s'."%movie)
		if not distance or not callable(distance):
			raise KeyError("Unknow or unprogrammed distance metric '%s'."%distance)
		items = {}
		for item in self.movies: #遍历字典的key（item此处为key，movieID）
			if item == movie:
				continue
			items[item] = distance(item,movie,prefs='movies')
		if n:
			return heapq.nlargest(n,item.items(),key=itemgetter(1))
		return items

