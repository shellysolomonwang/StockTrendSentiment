#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" Provides a merged dataset for the stock to be analyzed.

Clean datasets from 4 sources filtered by the desire stock and merge them into one.
"""


# In[1]:


import pandas as pd
import numpy as np
import nltk
nltk.download("vader_lexicon")
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
import yfinance as yf
#stock = 'TSLA'


# In[2]:


__author__ = "Tianrui Wang"
__copyright__ = "Copyright 2020, DS4A Women's Summit"
__email__ = "shellysolomonwang@gmail.com"


# In[21]:


def getVader(stock):
    """
    return df with vader scores for each comment/post
    
    """
    comments_posts = pd.read_csv('./data/posts_and_comments.csv', index_col=0)
    comments_posts.date_created = pd.to_datetime(comments_posts.date_created)
    comments_posts['weekday'] = pd.to_datetime(comments_posts.date_created).dt.weekday
    # 0 - Monday,4 - Friday, 5 - Sat, 6 - Sunday
    comments_posts['datetime']= comments_posts.date_created
    # change Satureday posts to Friday
    comments_posts.loc[comments_posts.weekday==5,'datetime'] = comments_posts[comments_posts.weekday==5].date_created - datetime.timedelta(days=1)
    # mark Sunday posts to Satureday
    comments_posts.loc[comments_posts.weekday==6,'datetime'] = comments_posts[comments_posts.weekday==6].date_created - datetime.timedelta(days=2)
    comments_posts = comments_posts[comments_posts.tickers.str.contains(stock)]

    # initialize VADER 
    vad = SentimentIntensityAnalyzer()
    # update with new words
    new_words = {"moon":2, "bear":-2, "printer":2, "bull":2, "drill":2, "put":-2, "call":2, "long":2, "short":-2, "up":2, "down":-2, "green": 2, "red":-2, "drop":-0.5, "rocket":1.5}
    vad.lexicon.update(new_words)
    
    # iterate through the whole dataframe and calculate compound vad score
    vad_compound = []
    for index, row in comments_posts.iterrows():
        msg = row["body"]
        vad_compound.append(vad.polarity_scores(msg)["compound"])
    comments_posts["vader_score"] = vad_compound
    comments_posts.drop(columns = ['weekday'], inplace = True)
    return comments_posts

def cleanVader(stock):
    """
    """
    comments = getVader(stock)
    # comments['datetime'] = pd.to_datetime(comments.timestamp).dt.date
    vader_df = pd.DataFrame(columns = ['vader_mean','vader_std', 'comment_count','pos_count','neg_count'])
    vader_df.vader_mean = comments.groupby('datetime').vader_score.mean()
    vader_df.vader_std = comments.groupby('datetime').vader_score.std().fillna(0)
    vader_df.comment_count = comments.groupby('datetime').vader_score.count()
    vader_df.pos_count = comments.groupby('datetime')['vader_score'].apply(lambda x: (x>0).sum())
    vader_df.neg_count = comments.groupby('datetime')['vader_score'].apply(lambda x: (x<0).sum())
    
    pos_mean = comments[comments.vader_score > 0].groupby('datetime').mean().rename(columns={'vader_score':'pos_mean'})
    neg_mean = comments[comments.vader_score < 0].groupby('datetime').mean().rename(columns={'vader_score':'neg_mean'})
    
    pos_sum = comments[comments.vader_score > 0].groupby('datetime').sum().rename(columns={'vader_score':'pos_sum'})
    neg_sum = comments[comments.vader_score < 0].groupby('datetime').sum().rename(columns={'vader_score':'neg_sum'})
    
    vader_df = vader_df.join(pos_mean, how='outer').join(neg_mean,how='outer').join(pos_sum, how='outer').join(neg_sum, how='outer').fillna(0)
    
    return vader_df

def getCommentsVader(stock):
    """
    return a comments_df with vader scores for each comment
    """
    # read comments + filter for the stock
    comments_df =pd.read_csv('./data/comment_tickers_mentioned_python_new.csv')
    comments_df = comments_df[comments_df.tickers.str.contains(stock)]
    
    # initialize VADER 
    vad = SentimentIntensityAnalyzer()
    
    # iterate through the whole dataframe and calculate compound vad score
    vad_compound = []
    for index, row in comments_df.iterrows():
        msg = row["body"]
        vad_compound.append(vad.polarity_scores(msg)["compound"])
    comments_df["vader_score"] = vad_compound
    comments_df.drop(columns = ['weekday','type'], inplace = True)
    return comments_df

def cleanCommentsVader(stock):
    """
    """
    comments = getCommentsVader(stock)
    comments['datetime'] = pd.to_datetime(comments.timestamp).dt.date
    vader_df = pd.DataFrame(columns = ['vader_mean','vader_std', 'comment_count','pos_count','neg_count'])
    vader_df.vader_mean = comments.groupby('datetime').vader_score.mean()
    vader_df.vader_std = comments.groupby('datetime').vader_score.std().fillna(0)
    vader_df.comment_count = comments.groupby('datetime').vader_score.count()
    vader_df.pos_count = comments.groupby('datetime')['vader_score'].apply(lambda x: (x>0).sum())
    vader_df.neg_count = comments.groupby('datetime')['vader_score'].apply(lambda x: (x<0).sum())
    
    pos_count = comments[comments.vader_score > 0].groupby('datetime').mean().rename(columns={'vader_score':'pos_mean'})
    neg_count = comments[comments.vader_score < 0].groupby('datetime').mean().rename(columns={'vader_score':'neg_mean'})
    vader_df = vader_df.join(pos_count, how='outer').join(neg_count,how='outer').fillna(0)
    return vader_df

def getPrice(stock):
    """
    return price_df for the stock with close & volume
    """
    # read df + filter stocks
    price_df = pd.read_csv('./data/top30_stock_price.csv', index_col=0, parse_dates=True)
    price_df = price_df[price_df.ticker == stock]
    return price_df

def cleanPrice(stock):
    """
    Note: NaN existed in result
    1. calculate close_T-1 (previous day closing price)
    2. calculate target variable (binary) of next day's return
        - 1 for positive return
        - 0 for negative return
    3. calculate returnPrev1 (1-day log return by closing price)
    """
    price_df = getPrice(stock)
    price_df['close_T-1'] = price_df.close.shift(periods =1)
    price_df['Y']= price_df.apply(lambda x: 1 if (x['close'] - x['close_T-1']) > 0 else 0, axis=1).shift(periods =-1)
    # returnPrev1 = ln(close_T / close_T-1)
    price_df['logreturnPrev1'] = np.log(price_df.close/price_df['close_T-1'])
    price_df['returnPrev1'] = price_df.close.pct_change(periods = 1)
    price_df['returnPrev5'] = price_df.close.pct_change(periods = 5)
    return price_df.drop(columns = ['ticker'])

def getPriceYahoo(stock):
    """
    """
    price_df = yf.download(stock,'2018-04-02')[['Open', 'Close', 'Volume']]
    return price_df

def cleanPriceYahoo(stock):
    price_df = getPriceYahoo(stock)
    price_df['close_T-1'] = price_df.Close.shift(periods =1)
    price_df['Y']= price_df.apply(lambda x: 1 if (x['Close'] - x['close_T-1']) > 0 else 0, axis=1).shift(periods =-1)
    price_df['logReturnClosePrev1'] = np.log(price_df.Close/price_df['close_T-1'])
    price_df['returnClosePrev1'] = price_df.Close.pct_change(periods = 1)
    price_df['returnClosePrev5'] = price_df.Close.pct_change(periods = 5)
    price_df['returnOpenPrev1'] = price_df.Open.pct_change(periods = 1)
    price_df['returnOpenPrev5'] = price_df.Open.pct_change(periods = 5)
    return price_df
    
def getVIX():
    vix_df = pd.read_csv('./data/vixcurrent.csv', skiprows=1, index_col=0)
    vix_df.index = pd.to_datetime(vix_df.index, format = '%m/%d/%Y')
    vix_df['VIX_Close_T-1'] = vix_df['VIX Close'].shift(periods=1)
    vix_df['vix_returnPrev1'] = np.log(vix_df['VIX Close']/vix_df['VIX_Close_T-1'])
    return vix_df

def getPopularity(stock):
    popularity_df = pd.read_csv('./data/popularity_cleaned.csv')
    popularity_df = popularity_df[popularity_df.ticker ==stock]
    return popularity_df.set_index('datetime').drop(columns = ['ticker']).add_prefix('popularity_')

def merge(stock):
    """
    main function
    return cleaned df for model
    """
    # get & clean comments, return vader matrix
    vader = cleanVader(stock)
    
    # get & clean price
    price = cleanPrice(stock)
    
    # get vix
    vix = getVIX()
    
    # get popularity
    popularity = getPopularity(stock)
    
    return popularity.join(price, how='inner').join(vix, how='left').join(vader, how='left').fillna(0)

def mergeYahoo(stock):
    """
    main function
    return cleaned df for model
    """
    # get & clean comments, return vader matrix
    vader = cleanVader(stock)
    
    # get & clean price
    price = cleanPriceYahoo(stock)
    
    # get vix
    vix = getVIX()
    
    # get popularity
    popularity = getPopularity(stock)
    
    return popularity.join(price, how='inner').join(vix, how='left').join(vader, how='left').fillna(0)


# In[ ]:




