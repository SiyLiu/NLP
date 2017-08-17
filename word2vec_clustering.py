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
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import sklearn



#对每一条工单中的词语训练word2vec模型
case_file =open("D:\\data\\ByCase.txt",encoding = 'utf-8')
case_model = Word2Vec(LineSentence(case_file),min_count = 5,size = 100)
case_save = "D:\\data\\case_model.bin"
case_model.save(case_save) #保存训练好的模型以便下次调用
case_vector_save = "D:\\data\\vector.txt"
word_vectors = case_model.wv.syn0 # 训练好的词向量
case_model.wv.save_word2vec_foramt(case_vector_save) #把向量导出到文件中

case_model = gensim.models.Word2Vec.load(case_save)#调用存好的模型

#Kmeans聚类
#以K=20为例
km = KMeans(n_clusters = 20)
case_result = km.fit_predict(word_vectors)
word_map = dict(zip(case_model.wv.index2word,case_result))

#把聚类结果导出
case_write = open("D:\\data\\cluster_result.txt","w",encoding = 'utf-8')
for cluster in range(20):
    case_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(word_map.values())):
        if list(word_map.values())[i] ==cluster:
            case_write.write(list(word_map.keys())[i]+' ')
case_write.close()

# 以silhouette_score为标准看kmeans聚类结果 -- 结果很差，2组的时候silhouette最大
compare = []
for i in range(10,50):
    ikm = []
    km = KMeans(n_clusters = i)
    case_result = km.fit_predict(word_vectors)
    score = silhouette_score(word_vectors,case_result)
    ikm.append(i)
    ikm.append(score)
    compare.append(ikm)

#对word_vectors进行层级聚类
np.set_printoptions(precision = 5, suppress = True)
Z = linkage(word_vectors, 'ward') # ‘ward’方法使用的是离差平方和算法
c,coph_dist = cophenet(Z,pdist(word_vectors)) # cophenet()计算同表象相关系数来判断集群的性能值越接近于1，集群结果就更加完整的保存了实验样本间的实际距离
print(c)
#对于层级聚类的可视化
plt.figure(figsize = (50,10))
dendrogram(Z,leaf_rotation = 90., leaf_font_size =8)
plt.show()

#AP聚类
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

ap = AffinityPropagation(damping = 0.5,convergence_iter= 50).fit_predict(word_vectors)
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

#密度聚类
from sklearn.cluster import DBSCAN
dbscan= DBSCAN(eps = 1,min_samples = 100).fit_predict(word_vectors)
density_result = dict(zip(case_model.wv.index2word, dbscan))

key_write = open("D:\\data\\key_cluster.txt","w",encoding = 'utf-8')

for cluster in range(20):
    key_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(word_centroid_map.values())):
        if list(word_centroid_map.values())[i] ==cluster:
            key_write.write(list(word_centroid_map.keys())[i]+' ')
key_write.close()


#SOM聚类
##这个算法是我抄网上的，自己并不怎么明白。
import random
import math

X_train = np.zeros(word_vectors.shape)
for j in range(word_vectors.shape[1]):
    for i in range(word_vectors.shape[0]):
        X_train[i,j] = (word_vectors[i,j] - word_vectors[:,j].mean())/word_vectors[:,j].std()
input_layer = X_train      

class Som_simple_zybb():
    def __init__(self,category):
        self.input_layer = input_layer # 输入样本
        self.output_layer = [] # 输出数据
        self.step_alpha = 0.5 # 步长 初始化为0.5
        self.step_alpha_del_rate = 0.95 # 步长衰变率
        self.category = category # 类别个数
        self.output_layer_length = len(self.input_layer[0]) # 输出节点个数 2
        self.d = [0.0] * self.category
 
    # 初始化 output_layer
    def initial_output_layer(self):
        for i in range(self.category):
            self.output_layer.append([])
            for _ in range(self.output_layer_length):
                self.output_layer[i].append(random.randint(0,400))
 
    # som 算法的主要逻辑
    # 计算某个输入样本 与 所有的输出节点之间的距离,存储于 self.d 之中
    def calc_distance(self,a_input ):
        self.d = [0.0] * self.category
        for i in range(self.category):
            w = self.output_layer[i]
            # self.d[i] =
            for j in range(len(a_input)):
                self.d[i] += math.pow((a_input[j] - w[j]),2) # 就不开根号了
 
    # 计算一个列表中的最小值 ，并将最小值的索引返回
    def get_min(self,a_list):
        min_index = a_list.index(min(a_list))
        return min_index
 
    # 将输出节点朝着当前的节点逼近
    def move(self,a_input,min_output_index):
        for i in range(len(self.output_layer[min_output_index])):
            self.output_layer[min_output_index][i] = self.output_layer[min_output_index][i] + self.step_alpha * ( a_input[i] - self.output_layer[min_output_index][i] )
 
    # som 逻辑 (一次循环)
    def train(self):
        for a_input in self.input_layer:
            self.calc_distance(a_input)
            min_output_index = self.get_min(self.d)
            self.move(a_input,min_output_index)
 
    # 循环执行som_train 直到稳定
    def som_looper(self):
        generate = 0
        while self.step_alpha >= 0.0001: # 这样子会执行167代
            self.train()
            generate +=1
            print("代数:{0} 此时步长:{1} 输出节点:{2}".format(generate,self.step_alpha,self.output_layer))
            self.step_alpha *= self.step_alpha_del_rate # 步长衰减
category = 20
som_zybb = Som_simple_zybb(category)
som_zybb.initial_output_layer()
som_zybb.som_looper()
      
