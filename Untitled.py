#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# In[3]:


countries = ['KR']
df = pd.DataFrame()
df1 = pd.DataFrame()

for c in countries:
    path = c + '_youtube_trending_data.csv'
    df1 = pd.read_csv(path, parse_dates=['publishedAt','trending_date'])
    df1['country'] = c
    df = pd.concat([df, df1])

df.shape


# In[24]:


df


# In[4]:


df.info()


# In[5]:


for i in df.columns:
    print(i+":",len(df[str(i)].value_counts()))
    print("-------------------------------------")


# In[7]:


df.columns


# In[8]:


df.head()


# In[9]:


# Remove unused columns
df = df.drop(['thumbnail_link'], axis=1)
# trending_date column only needs date format
df.trending_date = df.trending_date.dt.date

start_date = df.trending_date.min()
end_date = df.trending_date.max()
print ('Date covered from %s to %s' % (start_date, end_date))
print ('No. of days covered: %d' % df.trending_date.nunique())


# In[10]:


# Make list of videos
v_col = ['video_id', 'title', 'publishedAt', 'channelId', 'channelTitle']
videos = df[v_col].drop_duplicates(subset='video_id', keep='last') # Treat video_id as unique

# Compute video-based statistics
video_stat = df.groupby('video_id').agg({'video_id':'count',
    'trending_date': ['nunique','min','max'],
                           'view_count': 'max',
                           'likes': 'max',
                           'dislikes': 'max',
                           'comment_count': 'max',
                           'country': ['unique','nunique']})

video_stat.columns = ['trending_count','days_trend', 'first_trend_date','last_trend_date','views','likes','dislikes',
                      'comments','country_list','country_count']
video_stat.reset_index(inplace=True)
video_stat.head()


# In[11]:


videos = videos.merge(video_stat, on='video_id')


# In[12]:


# Top 10 videos that are trending the most time
videos.sort_values('trending_count', ascending=False).head(10)[['title','channelTitle','trending_count','country_list']]


# In[13]:


# 10 Highest view videos
videos.sort_values('views', ascending=False).head(10)[['title','channelTitle','views','country_list']]


# In[ ]:


# 10 Highest view videos
videos.sort_values('views', ascending=False).head(10)[['title','channelTitle','views','country_list']]


# In[14]:


# Top 10 most liked videos
videos.sort_values('likes', ascending=False).head(10)[['title','channelTitle','likes','country_list']]


# In[15]:


# Top 10 most dislike videos
videos.sort_values('dislikes', ascending=False).head(10)[['title','channelTitle','dislikes','country_list']]


# In[16]:


# Top 10 videos by comments
videos.sort_values('comments', ascending=False).head(10)[['title','channelTitle','comments','country_list']]


# In[17]:


# Trending videos with lowest views
videos.sort_values('views').head(10)[['title','channelTitle','views','country_list']]


# In[18]:


# Oldest videos that become trending?
videos.sort_values('publishedAt').head(10)[['title','channelTitle','publishedAt','first_trend_date','country_list']]


# In[19]:


channel_stat = videos.groupby('channelId').agg({'video_id':'count',
                               'views': ['sum','mean'],
                           'days_trend': 'sum'})
channel_stat.columns = ['no of videos','total views', 'average views', 'total days trending']
channel_stat.reset_index(inplace=True)

channel_names = df[['channelId','channelTitle']].drop_duplicates(subset='channelId', keep='last')
channel_stat = channel_stat.merge(channel_names, on='channelId')


# In[20]:


# Channels with the greatest number of trending videos
channel_stat.sort_values('no of videos', ascending=False).head(10)[['channelTitle','no of videos']]


# In[21]:


# Channels with highest total views
channel_stat.sort_values('total views', ascending=False).head(10)[['channelTitle','total views', 'no of videos']]


# In[22]:


channel_stat['average views'] = channel_stat['average views'].astype('int64')


# In[23]:


# Channels with highest average views
channel_stat.sort_values('average views', ascending=False).head(10)[['channelTitle','average views', 'no of videos']]


# In[ ]:




