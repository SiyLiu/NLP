# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:12:06 2017

@author: l81024167
"""
from gensim.models import Word2Vec
import gensim.models as md
import jieba
from sklearn.cluster import KMeans
from gensim.models.word2vec import LineSentence
import numpy as np
import matplotlib as nlp
import gensim
import scipy as sp
import scipy.sparse.linalg as lg


import networkx as nx

f = "D:\\data\\ByDay.txt"

class CorpusYielder(object):
    def __init__(self, path):
        self.path = path
    def __iter__(self):
        for line in open(self.path,'r',encoding = 'utf-8'):
            yield line.split(' ')

Model = Word2Vec()
Model.build_vocab(f)

sentences = md.doc2vec.TaggedLineDocument(f)
model = md.Doc2Vec(sentences,size = 100,window =3)

f_save = "D:\\data\\all_model.txt"
model.save(f_save)
        
num_clusters = 50
km_day=KMeans(n_clusters = 50)


result = km_day.fit_predict(model.wv.syn0)
day_centroid_map = dict(zip(model.wv.index2word,result))

day_write = open("D:\\data\\cluster_result_day.txt","w",encoding = 'utf-8')

for cluster in range(50):
    day_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(day_centroid_map.values())):
        if list(day_centroid_map.values())[i] ==cluster:
            day_write.write(list(day_centroid_map.keys())[i]+' ')
day_write.close()


    
    
model = gensim.models.Word2Vec.load(f_save)


case_file =open("D:\\data\\ByCase.txt",encoding = 'utf-8')

case_model = Word2Vec(LineSentence(case_file),min_count = 5,size = 100)
case_save = "D:\\data\\case_model.bin"
case_vector_save = "D:\\data\\vector.txt"
word_vectors = case_model.wv.syn0
case_model.wv.save_word2vec_foramt(case_vector_save, binary=False)

case_model = gensim.models.Word2Vec.load(case_save)
num_clusters = 50
km = KMeans(n_clusters = 20)
case_result = km.fit_predict(word_vectors)
word_centroid_map = dict(zip(case_model.wv.index2word,case_result))



case_write = open("D:\\data\\cluster_result2.txt","w",encoding = 'utf-8')

for cluster in range(20):
    case_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(word_centroid_map.values())):
        if list(word_centroid_map.values())[i] ==cluster:
            case_write.write(list(word_centroid_map.keys())[i]+' ')
case_write.close()

            
case_model.save(case_save)

case_open = open("D:\\data\\ByCase.txt",encoding = 'utf-8')
case_write = open("D:\\data\\cluster_result2.txt","w",encoding = 'utf-8')
for i in range(num_clusters-1):
    for linenum, eachline in enumerate(case_open.readlines()):
        if linenum >= len(case_result):
            break
        if result[linenum] ==i:
            case_write.write(eachline+'\n')
    case_write.write(str(i)+'-----------hello-------------\n')
    
# 以silhouette_score为标准看kmeans聚类结果 -- 结果很差，2组的时候silhouette最大
from sklearn.metrics import silhouette_score

compare = []
for i in range(10,50):
    ikm = []
    km = KMeans(n_clusters = i)
    case_result = km.fit_predict(word_vectors)
    score = silhouette_score(word_vectors,case_result)
    ikm.append(i)
    ikm.append(score)
    compare.append(ikm)

# 层级聚类
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np



np.set_printoptions(precision = 5, suppress = True)
Z = linkage(word_vectors, 'ward')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c,coph_dist = cophenet(Z,pdist(word_vectors))
plt.figure(figsize = (50,10))
dendrogram(Z,leaf_rotation = 90., leaf_font_size =8)
plt.show()

#AP聚类
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

ap = AffinityPropagation(convergence_iter= 50).fit_predict(word_vectors)
ap_result = dict(zip(case_model.wv.index2word,ap))

ap_case_write = open("D:\\data\\ap_result_case.txt","w",encoding = 'utf-8')

for cluster in range(max(ap)+1):
    ap_case_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(ap_result.values())):
        if list(ap_result.values())[i] ==cluster:
            ap_case_write.write(list(ap_result.keys())[i]+' ')
ap_case_write.close()

#Spectral 聚类

from sklearn.cluster import SpectralClustering
spc = SpectralClustering(20).fit_predict(word_vectors)
spc_result = dict(zip(case_model.wv.index2word,spc))

spc_case_write = open("D:\\data\\spc_result_case.txt","w",encoding = 'utf-8')

for cluster in range(20):
    spc_case_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(spc_result.values())):
        if list(spc_result.values())[i] ==cluster:
            spc_case_write.write(list(ap_result.keys())[i]+' ')
spc_case_write.close()

#TextRank key words
from textrank4zh import TextRank4Keyword, TextRank4Sentence
text = "桌面云上不了了，怎么办？"
tr4w = TextRank4Keyword()
tr4w.analyze(text = text,lower = True, window = 2)