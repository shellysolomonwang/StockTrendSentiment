#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import numpy as np 
from mergeStock import merge, mergeYahoo
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def dataProcessing(stock,scaler_choice='MinMax', split=0.8, features='customized'):
	df = mergeYahoo(stock)
    # define input variable
	df['leading_sentiment'] = (df.pos_sum.abs() > df.neg_sum.abs())
	if features =='customized':
        X = ['popularity_daily_avg','popularity_daily_change', 
             'Open', 'Close','Volume', 'returnClosePrev1', 'returnClosePrev5', 'returnOpenPrev1', 'returnOpenPrev5',
             'vix_returnPrev1',
             'vader_mean', 'vader_std','comment_count', 'pos_mean', 'neg_mean','leading_sentiment']
    if features =='all':
        X = df.columns.tolist()
        X.remove('Y')
    # define target variable
    Y = ['Y']
    df[Y] = df[Y].astype(int)
    
    # train test split 
    train = df[:int(len(df)*split)]
    test = df[int(len(df)*split):]
    
    # MinMax Scaler
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    if scaler_choice == 'MinMax':
        scaler = MinMaxScaler()
        train[X] = scaler.fit_transform(train[X])
        test[X] = scaler.transform(test[X])
    if scaler_choice == 'Smooth':
        scaler = MinMaxScaler()
            # Train the Scaler with training data & smooth data
        smoothing_window_size = 90
        for di in range(0,len(train),smoothing_window_size):
            try:
                scaler.fit(train[X][di:di+smoothing_window_size])
                train.loc[train.index[di]:train.index[di+smoothing_window_size],X] = scaler.transform(train.loc[train.index[di]:train.index[di+smoothing_window_size],X])
            except:
                break
        # normalize the last bit
        print(di)
        scaler.fit(train[X][di:])
    #         train[X][di+smoothing_window_size:] = scaler.transform(train[X][di+smoothing_window_size:])
        train.loc[train.index[di]:,X] = scaler.transform(train.loc[train.index[di]:,X])
        test[X] = scaler.transform(test[X])
        print('finished!')
    if scaler_choice == 'Standard':
        scaler = StandardScaler()
        train[X] = scaler.fit_transform(train[X])
        test[X] = scaler.transform(test[X])
    return train[X], train[Y].values, test[X], test[Y].values

def getTop30():
	df = pd.read_excel('/Users/shelly/Google Drive/DS4A - Team 23/data/top_30_summary_final.xlsx')
	stock_list = df.ticker.tolist()
	return stock_list

def main(stock):
	train_X, train_Y, test_X, test_Y = dataProcessing(df)
	model = MLPClassifier('activation': 'relu', 'hidden_layer_sizes': (32,), 'max_iter': 700, 'random_state': 42, 'solver': 'adam')
	model.fit(train_X, train_Y)
	y_true, y_pred = test_Y, model.predict(test_X)
	print(stock, accuracy_score(y_true, y_pred))
	return

stock_list = getTop30()
for stock in stock_list:
	main(stock)