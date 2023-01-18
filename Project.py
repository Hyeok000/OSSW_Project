#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


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


# In[3]:


#카테고리 ID를 통해 카테고리 이름 추가
import plotly.express as px
import json

with open("US_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['categoryId'].map(cat_dict)


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.columns


# In[8]:


#사용하지 않는 열 삭제
df = df.drop(columns=['video_id', 'publishedAt', 'channelId',
       'categoryId', 'trending_date', 'view_count', 'likes',
       'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled',
       'ratings_disabled'])


# In[9]:


df.columns


# In[10]:


df.info()


# In[11]:


#null 값의 개수 확인
df.isnull().sum()


# In[12]:


#title이 중복되는 행 삭제
df = df.drop_duplicates(subset='title', keep="first")


# In[13]:


df.info()


# In[14]:


df.isnull().sum()


# In[15]:


#NaN 값을 공백으로 채움
df = df.fillna('')


# In[16]:


#null 값 존재X
df.isnull().sum()


# In[17]:


df.info()


# In[18]:


#제목,채널 이름, 태그, 설명, 국가, 카테고리를 개요로 그룹화
df['overview'] = df['title']+" "+df['channelTitle']+" "+ df['tags']+" "+df['description']+" "+df['country']+" "+df['category_name'] 


# In[19]:


df.head()


# In[20]:


#채널 이름, 태그. 설명, 국가, 카테고리 열 삭제
df.drop(columns=['channelTitle', 'tags', 'description', 'country', 'category_name'], inplace=True)
df.head()


# In[21]:


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


# In[22]:


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
stop_words.extend(['com', 'https', 'ly', 'http', 'www', 'http_bit', 'youtube', 'bit'])

# stop_words.extend(['']) #extend existing stop word list if needed
def remove_stopwords(texts):
 return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Building  the bigram 
bigram = gensim.models.Phrases(all_words, min_count=5, threshold=10) 

bigram_mod = gensim.models.phrases.Phraser(bigram)
def create_bigrams(texts):
 return [bigram_mod[doc] for doc in texts]

# Lemmatization(명사, 형용사, 동사, 부사 표제어 추출)
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
 texts_out = []
 for sent in texts:
  doc = nlp(" ".join(sent)) 
  texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
 return texts_out


# In[23]:


all_words_nostops = remove_stopwords(all_words)
all_words_bigrams = create_bigrams(all_words_nostops)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
data_lemmatized = lemmatization(all_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])


# In[24]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Filter out tokens that appear in only 1 documents and appear in more than 90% of the documents
id2word.filter_extremes(no_below=2, no_above=0.9) #2개의 이상의 문서들, 0.9이상의 문서들

print(id2word[1])

# Create Corpus(말뭉치 생성)
texts = data_lemmatized

print(texts[1])

# Term Document Frequency(단어 출현 횟수)
corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[1])


# In[25]:


# Building LDA model for 10 topics
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=100, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=20,
                                       per_word_topics=True)


# In[26]:


vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)


# In[27]:


# Printing the Keywords in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[62]:


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


# In[29]:


df['Doc'] = doc_number
df['Topic'] = topic_number
df['Probability'] = prob
#df.to_csv("doc_topic_matrix.csv", index=False)


# In[30]:


df.info()


# In[31]:


df.head(10)


# In[32]:


df.head()


# In[387]:


import random

def recommend_by_title(title, df):
    
    title = title.split(',')
    title_list = [i.strip() for i in title]
    
    recommended = []
    top10_list = []
    
    #제목을 모두 소문자로 변경
    title_list = [i.lower() for i in title_list]
    df['overview'] = df['overview'].str.lower()


    try:
        sam = df[df['overview'].map(lambda x: all(word in x for word in title_list))]#개요 중에 키워드가 포함되는 샘플들
        #print(sam['title'].count())
        #print("2:",sam)
        sam = sam.sample(n=1)
        #print("3 :", sam)
    except ValueError:
        print("검색 불가")
        return
    
    topic_number = sam.Topic.values
    doc_number = sam.Doc.values
    
    output = df[df['Topic']==topic_number[0]].sort_values('Probability', ascending=False).reset_index(drop=True)
    index = output[output['Doc']==doc_number[0]].index[0]
   
    top10_list += list(output.iloc[index-5:index].index)
    top10_list += list(output.iloc[index+1:index+6].index)
    
    output['title'] = output['title'].str.title()
    
    for each in top10_list:
        recommended.append(output.iloc[each].title)
        
    return recommended


# In[34]:


df.info()


# In[64]:


#recommend_by_title('our first family intro!!', df)
recommend_by_title('smartphone', df)


# In[78]:


countries = ['KR']

df2 = pd.DataFrame()
df3 = pd.DataFrame()

for c in countries:
    path = c + '_youtube_trending_data.csv'
    df3 = pd.read_csv(path, parse_dates=['publishedAt','trending_date'])
    df3['country'] = c
    df2 = pd.concat([df2, df3])

#df = pd.read_csv('KR_youtube_trending_data.csv')
df2.info()


# In[79]:


with open("KR_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df2['category_name'] = df2['categoryId'].map(cat_dict)


# In[80]:


df2.info()


# In[81]:


df2.head()


# In[82]:


df2.shape


# In[83]:


df2.columns


# In[84]:


#사용하지 않는 열 삭제
df2 = df2.drop(columns=['video_id', 'publishedAt', 'channelId',
       'categoryId', 'trending_date', 'view_count', 'likes',
       'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled',
       'ratings_disabled'])


# In[86]:


df2.columns


# In[87]:


df2.info()


# In[88]:


#null 값의 개수 확인
df2.isnull().sum()


# In[89]:


#title이 중복되는 행 삭제
df2 = df2.drop_duplicates(subset='title', keep="first")


# In[91]:


df2.info()


# In[92]:


df2.isnull().sum()


# In[93]:


#NaN 값을 공백으로 채움
df2 = df2.fillna('')


# In[95]:


#null 값 존재X
df2.isnull().sum()


# In[96]:


df2.info()


# In[97]:


#제목,채널 이름, 태그, 설명, 국가, 카테고리를 개요로 그룹화
df2['overview'] = df2['title']+" "+df2['channelTitle']+" "+ df2['tags']+" "+df2['description']+" "+df2['country']+" "+df2['category_name'] 


# In[98]:


df2.head()


# In[99]:


#채널 이름, 태그. 설명, 국가, 카테고리 열 삭제
df2.drop(columns=['channelTitle', 'tags', 'description', 'country', 'category_name'], inplace=True)
df2.head()


# In[100]:


df2['overview']= df2['overview'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]',' ',regex=True)


# In[101]:


df2['overview'].head()


# In[102]:


df2['overview'].replace({'': np.nan})
df2['overview'].replace(r'^\s*$', None, regex=True)


# In[103]:


df2['overview'].dropna(how='any', inplace=True)


# In[104]:


df2.head()


# In[105]:


print(df2['overview'].isnull().values.any()) 


# In[106]:


df2.info()


# In[107]:


all_words_kr = list(sentence_to_words(df2['overview']))


# In[110]:


len(all_words_kr)


# In[112]:


all_words_nostops_kr = remove_stopwords(all_words_kr)


# In[113]:


# Create Dictionary
id2word_kr = corpora.Dictionary(all_words_kr)

# Filter out tokens that appear in only 1 documents and appear in more than 90% of the documents
id2word_kr.filter_extremes(no_below=2, no_above=0.9) #2개의 이상의 문서들, 0.9이상의 문서들

print(id2word_kr[1])

# Create Corpus(말뭉치 생성)
texts_kr = all_words_kr

print(texts_kr[1])

# Term Document Frequency(단어 출현 횟수)
corpus_kr = [id2word_kr.doc2bow(text) for text in texts_kr]

print(corpus_kr[1])


# In[114]:


# Building LDA model for 10 topics
lda_model_kr = gensim.models.LdaMulticore(corpus=corpus_kr,
                                       id2word=id2word_kr,
                                       num_topics=100, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=20,
                                       per_word_topics=True)


# In[116]:


vis_kr = pyLDAvis.gensim_models.prepare(lda_model_kr, corpus_kr, id2word_kr)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis_kr)


# In[121]:


# Printing the Keywords in the 10 topics
pprint(lda_model_kr.print_topics())
doc_lda_kr = lda_model_kr[corpus_kr]


# In[122]:


doc_number_kr , topic_number_kr, prob_kr = [], [], []
print(lda_model_kr.get_document_topics(corpus_kr))

for n in range(len(df2)):
    get_document_topics = lda_model_kr.get_document_topics(corpus_kr[n])
    doc_number_kr.append(n)
    sorted_doc_topics = Sort_Tuple(get_document_topics)
    #print(sorted_doc_topics)
    #i = sorted_doc_topics[0][0]
    try:
       # if not sorted_doc_topics[0][0]:
            #print('Null')
           # topic_number.append((''))
       # else:
        topic_number_kr.append(sorted_doc_topics[0][0])
    except IndexError:
        topic_number_kr.append((''))
        #continue
    #print(n)
    #print(topic_number)
    try:
        prob_kr.append(sorted_doc_topics[0][1])
    except IndexError:
        prob_kr.append((''))


# In[123]:


df2['Doc'] = doc_number_kr
df2['Topic'] = topic_number_kr
df2['Probability'] = prob_kr


# In[124]:


df2.info()


# In[125]:


df2.head(10)


# In[126]:


df = pd.concat([df, df2])


# In[128]:


df


# In[393]:


recommend_by_title('bts, 방탄소년단', df)


# In[319]:


from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# In[388]:


tk = Tk()
tk.title("Youtube Recommendation")
tk.geometry("800x400+100+100")


# In[389]:


#label = Label(window, text='')
global lists
global listbox
global sbar

#sbar = Scrollbar(tk)
#sbar.pack(side='bottom', fill='x')
listbox = Listbox(tk, height = 0, selectmode = "extended", width = 300)#, xscrollcommand = sbar.set)

def getTextInput():
    result=textbox.get()
    keywords(result)


def keywords(result):
    lists = []
    listbox.delete(0, END)
    lists = recommend_by_title(result, df)
    
    global count
    count = 0
    
    try:
        for i in range(len(lists)):
            listbox.insert(i+1, lists[i])
        listbox.pack()
    except TypeError:
        listbox.insert(1, "키워드 오류")

textbox = Entry(tk, width=40, textvariable=StringVar())
button = Button(tk,text='확인',font=('맑은 고딕',11,'bold'),bg="green",fg='white',width=4, command=getTextInput)

textbox.pack()
button.pack()


# In[390]:


tk.mainloop()


# In[ ]:

