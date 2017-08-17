# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:54:01 2017

@author: l81024167
"""
import imp
from sklearn.neighbors import NearestNeighbors
imp.reload(jieba)
imp.reload(jieba.analyse)

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
import jieba.posseg as psg
from nltk import *
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import HashingVectorizer 
from sklearn.cluster import KMeans 
from stop_words import get_stop_words
#import re
#from wordcloud import WordCloud
from gensim import corpora, models, similarities
import lda
import xlwt
import re


domain_open = 'D:\\data\\domain2.txt'
synonym_file = 'D:\\data\\synonym.xlsx'

synonym = pd.read_excel(synonym_file,header = None)
synonym = dict(zip(synonym[0],synonym[1]))

domainwords_file = pd.read_csv("D:\\data\\domainwords.txt",sep="\t",header = None)
domainwords = list(domainwords_file[0])
f = open(domain_open, "w", encoding = 'utf-8')
for item in domainwords:
    f.write(str(item)+'\n')
f.close()


jieba.load_userdict(domain_open)
jieba.analyse.set_stop_words('D:\\data\\newstop.txt')

file='D:\data\incidents_201612.xlsx'
IT= pd.read_excel(file) #Read the data into a dataframe
IT.info() #Check basic information of the dataframe

             # 169391 rows 43 columns

TITLE = list(IT.TITLE)
#删除相邻重复项
def dup_del(l):
    i =1
    while i <len(l):
        if l[i]==l[i-1]:
            l.pop(i)
        else:
            i+=1
    return l

##利用analyse.extract_tags提取每个工单中的关键词
TITLE_no_dup = pd.Series(dup_del(TITLE))
case_keys = TITLE_no_dup.map(analyse.extract_tags) 

write_case_key = 'D:\\data\\ByCaseKey.txt'
b = open(write_case_key, "w", encoding = 'utf-8')
for item in case_keys:
    b.write(' '.join(item)+'\n')
b.close()

case_key_open = open(write_case_key, "r", encoding = "utf-8")
##对于提取出的关键词进行word2vec向量化
case_key_model = Word2Vec(LineSentence(case_key_open),min_count = 100,size = 100)
case_key_save = "D:\\data\\case_key_model.bin"
key_vector_save = "D:\\data\\key_vector.txt"
keyword_vectors = case_key_model.wv.syn0
#case_key_model.wv.save_word2vec_foramt(key_vector_save)

##kmeans聚类
km = KMeans(n_clusters = 20)
case_key_result = km.fit_predict(keyword_vectors)
word_centroid_map = dict(zip(case_key_model.wv.index2word,case_key_result))

key_write = open("D:\\data\\key_cluster.txt","w",encoding = 'utf-8')

for cluster in range(20):
    key_write.write('\n'+str(cluster)+'-------------hello-------------\n')
    for i in range(len(word_centroid_map.values())):
        if list(word_centroid_map.values())[i] ==cluster:
            key_write.write(list(word_centroid_map.keys())[i]+' ')
key_write.close()

##利用TextRank提取关键词
from textrank4zh import TextRank4Keyword, TextRank4Sentence
text = "桌面云上不了了，怎么办？"
tr4w = TextRank4Keyword()
tr4w.analyze(text = text,lower = True, window = 2)