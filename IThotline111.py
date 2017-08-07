# -*- coding: utf-8 -*-
"""
Author: Siyuan Liu
Python Version: 3.6

"""
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

#Define domain dictionary
import imp 

domain_open = 'D:\\data\\domain.txt'
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

TITLE_no_dup = pd.Series(dup_del(TITLE))
            
#1 Description of interesting variables

#IT.columns

#Analysis by Date and Time
IT['OPEN DATE'].column=['OPEN_DATE']
IT.rename(columns = {"OPEN DATE":"OPEN_DATE", "OPEN TIME":"OPEN_TIME"},inplace = True)
IT['DAY'] = pd.to_datetime(IT.OPEN_DATE).dt.date
group_by_day = IT.groupby('DAY')
day_count = group_by_day.count().ID



IT.TIME =pd.to_datetime(IT.OPEN_TIME).dt.time
IT['HOUR']= pd.to_datetime(IT.OPEN_TIME).dt.hour
group_by_hour=IT.groupby('HOUR')
hour_count= group_by_hour.count().ID

IT_1201 = IT[IT.OPEN_DATE=="2016-12-01"]
group_by_hour_1201 = IT_1201.groupby('HOUR')
hour_count_1201 = group_by_hour.count().ID

#Analysis by Area
area = pd.Series.unique(IT['AREA'])
pd.Series.unique(IT['SUBAREA'])
group_by_area = IT.groupby(['DAY','AREA'])
area_count = group_by_area.count().ID
area_count =area_count.sort_values(ascending = False)

area_file = 'D:\\data\\area_day.txt"
for item in area_count

#Analysis by urgency
IT.URGENCY = IT.URGENCY.str[:1]
group_by_urgency = IT.groupby('URGENCY')
urgency_count= group_by_urgency.count().ID
group_urgency_day = IT.groupby(['URGENCY','DAY'])
urgency_day_count = group_urgency_day.count().ID

#Import stopwords
stopwords_file = open('D:\\data\\analysis\\stopwords.txt','r',encoding = 'utf-8')
stopwords=[]
for line in stopwords_file.readlines():
   stopwords.append(line.strip())
stopwords_file.close()
stopwords = stopwords +get_stop_words('en')+[' ','\n']+['almost','please','jqr']

## 写入停用词文件，为jieba.analyze做准备
stopwords_file = open('D:\\data\\newstop.txt','w',encoding='utf-8')
for word in stopwords:
    stopwords_file.write(str(word)+'\n')
stopwords_file.close()

## 分词 -- 结果为一个list
def cut_word(raw_data):
    cut_result = []    
    prbl_cl = list(raw_data)
    for entry in prbl_cl:
        word_cut= psg.lcut(entry)
        cut_result+=word_cut
    return cut_result

## 清洗分词结果


cut_all = cut_word(TITLE_no_dup)
cut_title = TITLE_no_dup.map(psg.lcut)




office  =list(set([w.word for w in cut_all if re.search('office',w.word)]))
outlook = list(set([w.word for w in cut_all if re.search('outlook',w.word)]))

def syn_rep(raw_data):
    for w in raw_data:
        if w.word in synonym.keys():
            w.word = synonym[w.word]
    return raw_data


def data_clean(raw_data):
    raw_data = [word.lower() for word in raw_data]
    raw_data = [word for word in raw_data if word not in stopwords ]
    raw_data = [word for word in raw_data if  word.isdigit()  ]
    raw_data = [word for word in raw_data if not re.search('^[a-z]$',word)==0]
    raw_data = [word for word in raw_data if not re.search('\.',word)]
    raw_data = [word for word in raw_data if not re.search('^[a-z]+[0-9]+',word) or word in office or word in outlook]
    raw_data = [word for word in raw_data if len(word)<=13 or word in office or word in outlook]
    return raw_data

def data_clean_psg(raw_data):
    raw_data = [w.word for w in raw_data if re.search('^n',w.flag) or re.search('^v',w.flag) or w.flag in ["eng",'x']]
    raw_data = [w.lower() for w in raw_data]
    raw_data = [w for w in raw_data if w not in stopwords ]
    raw_data = [w for w in raw_data if  w.isdigit() ==0 ]
    raw_data = [w for w in raw_data if not re.search('^[a-z]$',w)]
    raw_data = [w for w in raw_data if not re.search('\.',w)]
    raw_data = [w for w in raw_data if not re.search('^[a-z]+[0-9]+',w) or w in office or w in outlook]
    raw_data = [w for w in raw_data if not re.search('^[0-9]+[a-z]+',w) or w in office or w in outlook]
    raw_data = [w for w in raw_data if len(w)<=13 or w in office or w in outlook]
    
    return raw_data

daygroup = [group_by_day.get_group(x) for x in group_by_day.groups]

Question_day = []
for i in range(31):
    title_day = dup_del(list(daygroup[i].TITLE))
    Question_day.append(cut_word(title_day))

    
cut_day = pd.Series(Question_day)

no_syn_case = cut_title.map(syn_rep)
no_syn_day = cut_day.map(syn_rep)
no_syn_all = syn_rep(cut_all)


clean_all = data_clean_psg(no_syn_all)
clean_case = no_syn_case.map(data_clean_psg)
clean_day = no_syn_day.map(data_clean_psg)

##输出清洗好的数据

write_day_file = 'D:\\data\\ByDay.txt' 
c =open(write_day_file,encoding ='utf-8',mode = 'w')
for item in clean_day:
    c.write(' '.join(item)+'\n')
c.close()

write_case_file = 'D:\\data\\ByCase.txt'
b = open(write_case_file, "w", encoding = 'utf-8')
for item in clean_case:
    b.write(' '.join(item)+'\n')
b.close()

write_all_file = 'D:\\data\\All.txt'
a = open(write_all_file, encoding = 'utf-8',mode = 'w')
for item in clean_all:
    a.write(str(item)+'\n')
a.close()


# Topic extraction
stopwords_1='D:\\data\\newstop.txt'


###没用
jieba.analyse.set_stop_words(stopwords_1)
jieba.analyse.extract_tags(IT.TITLE[0])

IT_1201 = daygroup[0]
cut_1201 = cut_word(IT_1201.TITLE)
clean_1201 = data_clean(cut_1201)
freq_1201 = FreqDist(clean_1201)   

#计算文档词频矩阵
dictionary = corpora.Dictionary(clean_title)
corpus = [dictionary.doc2bow(doc) for doc in clean_title]

##构建LDA模型
dictionary = corpora.Dictionary(list(Question))
corpus = [dictionary.doc2bow(doc) for doc in Question]
tfidf = models.TfidfModel(corpus = corpus)
#
vector = tfidf[corpus[0]]
print(vector)
tfidf_corpus = tfidf[corpus]
corpora.MmCorpus.serialize('tfidf_corpus.mm',tfidf_corpus)

lda = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics = 5)
lda_corpus = lda[tfidf_corpus]

corpora.MmCorpus.serialize('lda_corpus.mm',lda_corpus)
print(lda.print_topics(20))

Lda = models.ldamodel.LdaModel
ldamodel = Lda(corpus, num_topics = 200, id2word = dictionary, passes=50)

##Vectorize
count_vec = TfidfVectorizer( tokenizer=lambda doc: doc, lowercase=False)
data_vec = count_vec.fit_transform(Question).toarray()
word = count_vec.get_feature_names() 

##聚类
#
clf = KMeans(n_clusters=5)
clf_result = clf.fit(data_vec)


#Frequency statistics

##每天前10位的词频
hot_day = []
for i in range(len(clean_day)):
        freq = FreqDist(clean_day[i])
        group_by_word = pd.Series(freq)
        sort_freq = group_by_word.sort_values(ascending = False)
        hot_day.append(sort_freq[:10])
hot_day_stat = pd.DataFrame(hot_day)
hot_day_stat.to_excel("D:\\data\\HotWords.xlsx")       

freq=FreqDist(cut_result)
cut_result = pd.Series(cut_result,name = 'cut_result')
word_len = pd.Series(cut_result).map(len)
word_len = pd.Series(word_len,name = 'word_len')
cut_len= pd.concat([cut_result,word_len],axis =1)
group_by_len=cut_len.groupby('word_len')
len_count = group_by_len.count()
group_by_cut =pd.Series(freq)
sort_freq = group_by_cut.sort_values(ascending = False)
appear_once=freq.hapaxes()
#jieba.analyse.extract_tags(cut_result,20)

#LDA model building 
nwordall = []
for t in IT.TITLE:
    words = psg.lcut(t)
    nword = []
    for w in words:
        if ((w.flag in ['n','v','a','eng']) and len(w.word)>1 and word_clean(w.word) ==1) :
            nword.append(w.word)
    nwordall.append(nword)



dic = corpora.Dictionary(Question)
corpus = [dic.doc2bow(text) for text in Question]
tfidf=models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lda = models.LdaModel(corpus_tfidf, id2word = dic, num_topics = 20)
corpus_lda = lda[corpus_tfidf]
corpora.Dictionayr



