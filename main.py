# -*- coding:utf-8 -*-
import time
from loadsMovielensReviews import *
data = relative_path('data\ml-100k\u.data')
item = relative_path('data\ml-100k\u.item')
model = Moiveslens(data,item)
print model.movies[814]['title'],model.movies[1642]['title'],\
model.movies[1599]['title'],model.movies[1500]['title']

#top 20 average_reviews() 评分最高的前n部电影（分别基于均值，基于贝叶斯均值）
# for mid,avg,num in model.top_rated(25):
# 	title = model.movies[mid]['title']
# 	print "[%0.3f average rating (%i reviews)] %s"%(avg,num,title)

#用户相似性：欧式距离
#print model.euclidean_distance(232,532)

#用户相似性:皮尔逊相关性
#print model.pearson_correlation(232,532)

#找出最相似的前n用户
# for item in model.similar_critics(232,'pearson',n=10):
# 	print "%4i:%0.3f"%item

#print model.similar_critics(232,'pearson')
#{1: 0.47796642007798934, 2: 0.05074987143382188, 3: 0.3713906763541038, 4: 0.5, 

#预测用户评分
# print model.predict_ranking(422,50,'euclidean')
# print model.predict_ranking(422,50,'pearson')
# 4.35413151722
# 4.3566797826

#预测用户对所有电影的评分
# for mid,rating in model.predict_all_rankings(578,'pearson',10):
# 	print "%0.3f:%s"%(rating,model.movies[mid]['title'])

#print model.shared_critics(1,2)

# for movie,similarity in model.similar_items(631,'pearson').items():
# 	print "%03.f:%s"%(similarity,model.movies[movie]['title'])