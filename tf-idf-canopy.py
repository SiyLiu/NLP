# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:09:22 2017

@author: l81024167
"""

import jieba  
import networkx as nx  
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.cluster import KMeans 
from sklearn.externals import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score


f = open("D:\\data\\ByCase.txt", encoding = 'utf-8')
bycase = []
for line in f.readlines():
    bycase.append(line)

f.close()

#词频矩阵
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(bycase)
word = vectorizer.get_feature_names()

#TF-IDF矩阵
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)


km_tf = KMeans(n_clusters = 50)
km_tf.fit_predict(tfidf)
clusters = km_tf.labels_.tolist()

case = {'case':bycase,'clusters':clusters}
bycase_clst=pd.DataFrame(case,index = [clusters])
groupbyc= bycase_clst.groupby('clusters')
groupbyc.count()
S
from sklearn.metrics.pairwise import pairwise_distances
from skleann.metrics.pairwise import cosine_similarity
distance = pairwise_distances.co(word_vectors)
distance = distance.reshape(distance.shape[0]**2)

import numpy as np

#def canopy(X,T1,T2,distance_metric = 'euclidean', filemap = None):
#    canopies = dict()
#    X1_dist = pairwise_distances(X,metric = distance_metric)
#    canopy_points = set(range(X.shape[0]))
#    while canopy_points:
#        point = canopy_points.pop()
#        i = len(canopies)
#        canopies[i] = {'c':point, 'points': list(np.where(X1_dist[point]<T2)[0])}
#        canopy_points = canopy_points.difference(set(np.where(X1_dist[point]<T1)[0]))
#    if filemap:
#        for canopy_id in canopies.keys():
#            canopy = canopies.pop(canopy_id)
#            canopy2 = {'c':filemap[canopy['c']], 'points': list()}
#            for point in canopy['points']:
#                canopy['points'].append(filemap[point])
#            canopies[canopy_id] = canopy2
#    return canopies

