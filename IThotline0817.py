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
import re

# 同义词
synonym_file = 'D:\\data\\synonym.xlsx' 
synonym = pd.read_excel(synonym_file,header = None)
synonym = dict(zip(synonym[0],synonym[1]))

#领域词库 由于jieba的词典和HanLp的词典格式不同，所以我只取了词语一列（第一列），没有要词频和词性
domainwords_file = pd.read_csv("D:\\data\\domainwords.txt",sep="\t",header = None)
domainwords = list(domainwords_file[0])
f = open(domain_open, "w", encoding = 'utf-8')
for item in domainwords:
    f.write(str(item)+'\n')
f.close()

# 导入领域词库
jieba.load_userdict(domain_open)

#导入工单数据
file='D:\data\incidents_201612.xlsx'
IT= pd.read_excel(file) #Read the data into a dataframe
IT.info() #Check basic information of the dataframe
          # 169391 rows 43 columns
TITLE = list(IT.TITLE)
#删除TITLE相邻重复项
def dup_del(l):
    i =1
    while i <len(l):
        if l[i]==l[i-1]:
            l.pop(i)
        else:
            i+=1
    return l

TITLE_no_dup = pd.Series(dup_del(TITLE))
            
#Description of interesting variables
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

#选取1201位样本看这一天每个小时工单情况
IT_1201 = IT[IT.OPEN_DATE=="2016-12-01"]
group_by_hour_1201 = IT_1201.groupby('HOUR')
hour_count_1201 = group_by_hour.count().ID

#Analysis by Area
area = pd.Series.unique(IT['AREA'])
pd.Series.unique(IT['SUBAREA'])
group_by_area = IT.groupby(['DAY','AREA'])
area_count = group_by_area.count().ID
area_count =area_count.sort_values(ascending = False)

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

## 写入停用词文件，为jieba.analyse做准备 （jieba.analyse用tfidf的方法提取关键词）
stopwords_file = open('D:\\data\\newstop.txt','w',encoding='utf-8')
for word in stopwords:
    stopwords_file.write(str(word)+'\n')
stopwords_file.close()

## 分词 -- 结果为一个list
#psg.lcut() 分词结果是词语+词性
def cut_word(raw_data):
    cut_result = []    
    prbl_cl = list(raw_data)
    for entry in prbl_cl:
        word_cut= psg.lcut(entry)
        cut_result+=word_cut
    return cut_result

#分词结果为一个list
cut_all = cut_word(TITLE_no_dup)

#分词结果为一个list中有len(title)个list
cut_title = TITLE_no_dup.map(psg.lcut)


#提取所有词中带有office 和outlook的词语
office  =list(set([w.word for w in cut_all if re.search('office',w.word)]))
outlook = list(set([w.word for w in cut_all if re.search('outlook',w.word)]))

#同义词替换
def syn_rep(raw_data):
    for w in raw_data:
        if w.word in synonym.keys():
            w.word = synonym[w.word]
    return raw_data



#清洗分词结果
#对于只有jieba.lcut()的结果进行清洗
def data_clean(raw_data):
    raw_data = [word.lower() for word in raw_data]
    raw_data = [word for word in raw_data if word not in stopwords ]
    raw_data = [word for word in raw_data if  word.isdigit()  ]
    raw_data = [word for word in raw_data if not re.search('^[a-z]$',word)==0]
    raw_data = [word for word in raw_data if not re.search('\.',word)]
    raw_data = [word for word in raw_data if not re.search('^[a-z]+[0-9]+',word) or word in office or word in outlook]
    raw_data = [word for word in raw_data if len(word)<=13 or word in office or word in outlook]
    return raw_data

#对于psg.lcut的结果进行清洗
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

#把工单按天分类
daygroup = [group_by_day.get_group(x) for x in group_by_day.groups]
Question_day = []
for i in range(31):
    title_day = dup_del(list(daygroup[i].TITLE))
    Question_day.append(cut_word(title_day))

#统计每天内area的情况
area_day=[]

for i in range(31):
    area =list(daygroup[i].AREA)
    freq = pd.Series(FreqDist(daygroup[i].AREA))
    groupbyarea = freq.sort_values(ascending = False)
    area_day.append(groupbyarea[:10])
hot_day_area = pd.DataFrame(area_day)
hot_day_area.to_excel("D:\\data\\HotArea10.xlsx")          
       
#对所有的分词结果、按天分词、按工单分词的结果进行去同义词处理
cut_day = pd.Series(Question_day)

no_syn_case = cut_title.map(syn_rep)
no_syn_day = cut_day.map(syn_rep)
no_syn_all = syn_rep(cut_all)

#清洗分词
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

#对词语长度的统计，为了确定应删除词语的长度
freq=FreqDist(clean_all)
clean_all = pd.Series(clean_all,name = 'cut_result')
word_len = pd.Series(clean_all).map(len)
word_len = pd.Series(word_len,name = 'word_len')
cut_len= pd.concat([cut_result,word_len],axis =1)
group_by_len=cut_len.groupby('word_len')
len_count = group_by_len.count()
group_by_cut =pd.Series(freq)
sort_freq = group_by_cut.sort_values(ascending = False)
appear_once=freq.hapaxes() #统计出现一次的词语

#统计每天词频最高的词语
hot_day = []
for i in range(len(clean_day)):
        freq = FreqDist(clean_day[i])
        group_by_word = pd.Series(freq)
        sort_freq = group_by_word.sort_values(ascending = False)
        hot_day.append(sort_freq[:10])
hot_day_stat = pd.DataFrame(hot_day)
hot_day_stat.to_excel("D:\\data\\HotWords5.xlsx")   


#构建LDA模型
word_dict = corpora.Dictionary(clean_day)
corpus_list = [word_dict.doc2bow(doc) for doc in clean_day]
lda = models.ldamodel.LdaModel(corpus=corpus_list, id2word=word_dict, num_topics = 10)

output_file = 'D:\\data\\lda_output.txt'
with open(output_file,'w') as f:
    for pattern in lda.show_topics():
        f.write("%s" % str(pattern))





