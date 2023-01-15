#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#한국은 나중에 추가 예정
countries = ['US']#, 'KR']

df = pd.DataFrame()
df1 = pd.DataFrame()

for c in countries:
    path = c + '_youtube_trending_data.csv'
    df1 = pd.read_csv(path, parse_dates=['publishedAt','trending_date'])
    df1['country'] = c
    df = pd.concat([df, df1])

#df = pd.read_csv('KR_youtube_trending_data.csv')
df.info()

#카테고리 ID를 통해 카테고리 이름 추가
import plotly.express as px
import json

with open("US_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['categoryId'].map(cat_dict)

df.info()

df.head()

df.shape

df.columns

#사용하지 않는 열 삭제
df = df.drop(columns=['video_id', 'publishedAt', 'channelId',
       'categoryId', 'trending_date', 'view_count', 'likes',
       'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled',
       'ratings_disabled'])

df.columns

df.info()

#null 값의 개수 확인
df.isnull().sum()

#title이 중복되는 행 삭제
df = df.drop_duplicates(subset='title', keep="first")

df.info()

df.isnull().sum()

#NaN 값을 공백으로 채움
df = df.fillna('')

#null 값 존재X
df.isnull().sum()

df.info()

#제목,채널 이름, 태그, 설명, 국가, 카테고리를 개요로 그룹화
df['overview'] = df['title']+" "+df['channelTitle']+" "+ df['tags']+" "+df['description']+" "+df['country']+" "+df['category_name'] 

df.head()

#채널 이름, 태그. 설명, 국가, 카테고리 열 삭제
df.drop(columns=['channelTitle', 'tags', 'description', 'country', 'category_name'], inplace=True)
df.head()

import warnings
warnings.filterwarnings("ignore")

# Importing modules
import pandas as pd
import numpy as np
import os
import re

# LDA Model
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from pprint import pprint
from gensim.models import CoherenceModel
import spacy
from nltk.corpus import stopwords

# Import the wordcloud library
from wordcloud import WordCloud
#!pip install pyLDAvis
#!python -m spacy download en

# Visualize the topics
import pyLDAvis
import pyLDAvis.gensim_models
import pickle 
import pyLDAvis

# Remove Non-english words
df['overview']= df['overview'].map(lambda x: re.sub('([^\x00-\x7F])+ ','', x))

# Tokenization(토큰화)
def sentence_to_words(sentences):
 for sentence in sentences:
  yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) # deacc=True removes punctuations and special characters

all_words = list(sentence_to_words(df['overview']))

# remove stopwords(불용어 제거)
import nltk
#nltk.download('stopwords')

stop_words = stopwords.words('english')

# stop_words.extend(['']) #extend existing stop word list if needed
def remove_stopwords(texts):
 return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Building  the bigram 
bigram = gensim.models.Phrases(all_words, min_count=5, threshold=10) 

bigram_mod = gensim.models.phrases.Phraser(bigram)
def create_bigrams(texts):
 return [bigram_mod[doc] for doc in texts]

# Lemmatization
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
 texts_out = []
 for sent in texts:
  doc = nlp(" ".join(sent)) 
  texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
 return texts_out

all_words_nostops = remove_stopwords(all_words)
all_words_bigrams = create_bigrams(all_words_nostops)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
data_lemmatized = lemmatization(all_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Filter out tokens that appear in only 1 documents and appear in more than 90% of the documents
id2word.filter_extremes(no_below=2, no_above=0.9)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Building LDA model for 10 topics
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=19, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

# Printing the Keywords in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

for i, topic_list in enumerate(doc_lda):
    if i==5:
        break
    print(i+1,'번째 문서의 topic 비율은',topic_list)

def Sort_Tuple(tup):  
    return(sorted(tup, key = lambda x: x[1], reverse = True))   

doc_number , topic_number, prob = [], [], []
print(lda_model.get_document_topics(corpus))
for n in range(len(df)):
    get_document_topics = lda_model.get_document_topics(corpus[n])
    doc_number.append(n)
    sorted_doc_topics = Sort_Tuple(get_document_topics)
    topic_number.append(sorted_doc_topics[0][0])
    prob.append(sorted_doc_topics[0][1])

df['Doc'] = doc_number
df['Topic'] = topic_number
df['Probability'] = prob
df.to_csv("doc_topic_matrix.csv", index=False)

def recommend_by_title(title, df):

    recommended = []
    top10_list = []
    
    title = title.lower()
    df['title'] = df['title'].str.lower()
    #print(df['title'])
    
    sam = df[df['title'].str.contains(title)].sample(n=1)
    #topic_number = df[df['title']==title].Topic.values
    topic_number = sam.Topic.values
    #print(topic_number)
    #doc_number = df[df['title']==title].Doc.values
    doc_number = sam.Doc.values
    #print(doc_number)
    
    output = df[df['Topic']==topic_number[0]].sort_values('Probability', ascending=False).reset_index(drop=True)

    index = output[output['Doc']==doc_number[0]].index[0]
    
    top10_list += list(output.iloc[index-5:index].index)
    top10_list += list(output.iloc[index+1:index+6].index)
    
    output['title'] = output['title'].str.title()
    
    for each in top10_list:
        recommended.append(output.iloc[each].title)
        
    return recommended

df.info()

recommend_by_title('korea', df)
