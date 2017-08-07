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
from nltk import *
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import re
from wordcloud import WordCloud
file='D:\data\incidents_201612.xlsx'
IT= pd.read_excel(file) #Read the data into a dataframe
IT.info() #Check basic information of the dataframe
             # 169391 rows 43 columns

#1 Description of interesting variables

#IT.columns

#Analysis by Date and Time
IT[['OPEN DATE']].column=['OPEN_DATE']
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
group_by_area = IT.groupby('AREA')
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

#Group titles by date
    
daygroup = [group_by_day.get_group(x) for x in group_by_day.groups]

Question = []
for i in range(31):
    Question.append(data_clean(cut_word(daygroup[i].TITLE)))
Question = pd.Series(Question)


IT_1201 = daygroup[0]
cut_1201 = cut_word(IT_1201.TITLE)
clean_1201 = data_clean(cut_1201)
freq_1201 = FreqDist(clean_1201)   



#Cut sentences and delete stopwords 
  
Q = IT.TITLE
Q = Q.map(jieba.lcut)

def del_stop(target):
    i = 0
    while i < len(target):
        target[i]=target[i].lower()
        if target[i] in stopwords:
            target.pop(i)
        elif target[i].isdigit ==1:
            target.pop(i)
        else:
            i+=1
    return target
        
Q_del = Q.map(del_stop)

#Vectorize
#count_vec = TfidfVectorizer( tokenizer=lambda doc: doc, lowercase=False, stop_words = "english")
#data_vec = count_vec.fit_transform(Q_del).toarray()

#Frequency statistics
def cut_word(raw_data):
    cut_result = []    
    prbl_cl = list(raw_data)
    for entry in prbl_cl:
        word_cut= jieba.lcut(entry)
        cut_result+=word_cut
    return cut_result

def data_clean(raw_data):
    office  =list(set([word for word in raw_data if re.search('office',word)]))
    outlook = list(set([word for word in raw_data if re.search('outlook',word)]))
    raw_data = [word.lower() for word in raw_data]
    raw_data = [word for word in raw_data if word not in stopwords ]
    raw_data = [word for word in raw_data if  word.isdigit() ==0 ]
    raw_data = [word for word in raw_data if not re.search('^[a-z]$',word)==0]
    raw_data = [word for word in raw_data if not re.search('\.',word)]
    raw_data = [word for word in raw_data if not re.search('^[a-z]+[0-9]+$',word) or word in office or word in outlook]
    raw_data = [word for word in raw_data if not re.search('^[0-9]+[a-z]',word) or word in office or word in outlook]
    raw_data = [word for word in raw_data if len(word)<=13 or word in office or word in outlook]
    return raw_data

    
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