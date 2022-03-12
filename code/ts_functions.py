import pandas as pd
import numpy as np
import requests
import json
import os
import time
import fred_msa
import datetime as dt
from datetime import date
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf

def ARIMA_pred(data, city, order=(2,2,0), trend='n'):
    
    # get data from one city and convert to array
    city_data = data[data.city == city].reset_index(drop=True)
    X = np.array(city_data.hpi)
    
    # set up df to store results
    results_df = city_data.set_index('date')['hpi'].to_frame()
    results_df = results_df.assign(pred_1=np.nan, pred_2=np.nan, pred_3=np.nan, pred_4=np.nan)
    
    # run predictions
    tscv = TimeSeriesSplit(n_splits=36, max_train_size=12, test_size=1)
    
    for train_index, test_index in tscv.split(X):
        X_train = X[train_index] # training set
        date = city_data.date[test_index[0]] # find date of first prediction
        city_model = ARIMA(X_train, order=order, trend=trend).fit() # train model
        preds = city_model.get_forecast(4) # predict 4 periods ahead
        pred_means = preds.predicted_mean
        pred_stds = preds.se_mean
        
        # save results
        results_df.loc[date, 'pred_1'] = pred_means[0]
        results_df.loc[date, 'pred_2'] = pred_means[1]
        results_df.loc[date, 'pred_3'] = pred_means[2]
        results_df.loc[date, 'pred_4'] = pred_means[3]
        
        results_df.loc[date, 'std_1'] = pred_stds[0]
        results_df.loc[date, 'std_2'] = pred_stds[1]
        results_df.loc[date, 'std_3'] = pred_stds[2]
        results_df.loc[date, 'std_4'] = pred_stds[3]
        results_df.loc[date, 'AIC'] = city_model.aic
        results_df.loc[date, 'pval'] = city_model.pvalues.mean()
        
    # shift predictions back to correct date
    results_df.pred_2 = results_df.pred_2.shift(1)
    results_df.pred_3 = results_df.pred_3.shift(2)
    results_df.pred_4 = results_df.pred_4.shift(3)
    
    results_df.std_2 = results_df.std_2.shift(1)
    results_df.std_3 = results_df.std_3.shift(2)
    results_df.std_4 = results_df.std_4.shift(3)
    
    # add in naive predictions (x_t+k = x_t for all k)
    results_df['naive_1'] = results_df.hpi.shift(1)
    results_df['naive_2'] = results_df.hpi.shift(2)
    results_df['naive_3'] = results_df.hpi.shift(3)
    results_df['naive_4'] = results_df.hpi.shift(4)

    return results_df[12:]

def evaluate_model(results_df, method='square'):
    
    # calculate diffs
    results_df = results_df.assign(diff_1=(results_df.pred_1-results_df.hpi)/results_df.hpi,
                  diff_2=(results_df.pred_2-results_df.hpi)/results_df.hpi,
                  diff_3=(results_df.pred_3-results_df.hpi)/results_df.hpi,
                  diff_4=(results_df.pred_4-results_df.hpi)/results_df.hpi,
                  diff_1_naive=(results_df.naive_1-results_df.hpi)/results_df.hpi,
                  diff_2_naive=(results_df.naive_2-results_df.hpi)/results_df.hpi,
                  diff_3_naive=(results_df.naive_3-results_df.hpi)/results_df.hpi,
                  diff_4_naive=(results_df.naive_4-results_df.hpi)/results_df.hpi)
    
    # list of difference columns
    diff_cols = [col for col in results_df.columns if 'diff' in col]                               
    
    # use square difference
    results_pos = results_df.copy()
    if method == 'square':
        results_pos[diff_cols] = results_pos[diff_cols] ** 2
        
    elif method == 'abs':
        results_pos[diff_cols] = np.abs(results_pos[diff_cols])
        
    else:
        raise ValueError('Choose a valid method for evaluation.')

    # calculate mse by city and lag value, caulculate winners
    mses = results_pos.groupby(['city', 'p', 'd', 'q'])[diff_cols].mean().reset_index()

    mses = mses.assign(improve_1=(mses.diff_1_naive-mses.diff_1),
                 improve_2=(mses.diff_2_naive-mses.diff_2),
                 improve_3=(mses.diff_3_naive-mses.diff_3),
                 improve_4=(mses.diff_4_naive-mses.diff_4))


    mses = mses.assign(win_1=mses.improve_1 > 0,
                 win_2=mses.improve_2 > 0,
                 win_3=mses.improve_3 > 0,
                 win_4=mses.improve_4 > 0)
    
    
    # best lag by win rate
    win_cols = [col for col in mses.columns if 'win' in col]
    win_rate = mses.groupby(['city', 'p', 'd', 'q'])[win_cols].sum().sum(axis=1)/mses.groupby(['city', 'p', 'd', 'q'])[win_cols].count().sum(axis=1)
    
    # best lag by minimum error
    pred_cols = [col for col in mses.columns if 'diff' in col and 'naive' not in col]
    avg_mse = mses.groupby(['city', 'p', 'd', 'q'])[pred_cols].mean().mean(axis=1)
    
    # best lag by max improvement
    improve_cols = [col for col in mses.columns if 'improve' in col]
    avg_improvement = mses.groupby(['city', 'p', 'd', 'q'])[improve_cols].mean().mean(axis=1)
    
    # best AIC, pval
    AIC = results_df.groupby(['city', 'p', 'd', 'q']).AIC.mean()
    pvals = results_df.groupby(['city', 'p', 'd', 'q']).pval.median()
    
    # return all three metrics
    return pd.concat([win_rate.to_frame('win'), avg_mse.to_frame('mse'), avg_improvement.to_frame('improvement'), AIC, pvals], axis=1)

def ARIMA_best_hypers(modeling_results, eval_results):
    '''
    GOAL - select the best hyperparameters balancing four selection criteria (win rate, AIC, improvement over naive, mse)
    INPUT - model output values, results from "evaluate_model" function
    OUTPUT - dataframe of results with best PDQ from each city
    '''
    
    results_normed = eval_results.groupby(['city']).transform(lambda x : (x - x.mean()) / x.std())
    
    # convert "negative" metrics
    low_cols = ['mse', 'AIC', 'pval']
    results_normed[low_cols] = -results_normed[low_cols]
    
    # aggregate scores
    agg_scores = results_normed.sum(axis=1).to_frame('agg_score').reset_index()
    
    # get best hypers (max agg score for each city)
    agg_scores = agg_scores.merge(agg_scores.groupby(['city']).agg_score.max(), on='city', how='inner')
    best_pdq = agg_scores[agg_scores.agg_score_x == agg_scores.agg_score_y].drop(columns = ['agg_score_x', 'agg_score_y'])
    
    # filter model results for best hypers
    best_results = modeling_results.merge(best_pdq, on=['city', 'p', 'd', 'q'], how='inner', validate='many_to_one')
    
    return best_results