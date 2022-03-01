# -*- coding: utf-8 -*-
"""
FRED API MSA Functions

This module contains a set of functions to 

Created on Tue Dec  7 11:21:03 2021

@author: tgule
"""

import pandas as pd
import numpy as np
import json
import os
import requests
import time
import datetime as dt
from datetime import date
import regex as re
import matplotlib.pyplot as plt
import seaborn as sns


########### Section 1 - Extracting City and State from MSA Names

def split_city(city):
    '''
    Takes out city name and other listed cities to isolate main city name fro MSA name
    '''
    split = re.split(',', city)
    
    city_stripped = ''
    
    try:
        city_stripped = str.replace(re.split('-|/', split[0])[0], ' County', '')
        
    except:
        pass
    
    return city_stripped

def extract_state(city):
    '''
    Extracts main state from list of city names
    '''
    
    split_state = re.split(',', city)
    
    state = ''
    
    try:
        state = str.strip(str.split(split_state[1], sep='-')[0])
        
    except:
        pass
    
    return state


########## Section 2 - Pulling Data with the FRED API

def get_state_ids(api_key):
    
    '''
    Gets a list of state ids from the FRED API. The parent category for all states has id '27281', 
    so getting all children returns the individual state ids
    '''
    
    params_location = {
    'api_key' : api_key,
    'category_id' : '27281',
    'file_type': 'json'
    }

    endpoint_location = 'https://api.stlouisfed.org/fred/category/children'
    
    response_location = requests.get(endpoint_location, params=params_location)
    state_ids = {s['name'] : s['id'] for s in json.loads(response_location.text)['categories']}
    
    return state_ids

def get_msa_cats(api_key, state_id):
    '''
    For a given FRED state id, get a list of FRED IDs for all Metro Statistical Areas in that state
    '''

    params_state = {
        'api_key' : api_key,
        'category_id' : state_id,
        'file_type': 'json'
    }

    endpoint_state = 'https://api.stlouisfed.org/fred/category/children'
    response_state = requests.get(endpoint_state, params=params_state)
    
    state_msas = {} # instantiate dictionary for MSAs to enure valid return statement
    
    cats = json.loads(response_state.text)['categories']
    
    if len(cats) > 0: # not all states/territories have MSAs
        
        # "categories call returns both counties and MSAs; we only want MSAs
        msa_cat = [s['id'] for s in json.loads(response_state.text)['categories']][1] 

        params_msacat = {
                'api_key' : api_key,
                'category_id' : msa_cat,
                'file_type': 'json'
            }

        endpoint_msacat = 'https://api.stlouisfed.org/fred/category/children'

        response_msacat = requests.get(endpoint_msacat, params=params_msacat)

        state_msas = {s['name'] : s['id'] for s in json.loads(response_msacat.text)['categories']}
    
    return state_msas

def get_county_cats(api_key, state_id):
    '''
    For a given FRED state id, get a list of FRED IDs for all Metro Statistical Areas in that state
    '''

    params_state = {
        'api_key' : api_key,
        'category_id' : state_id,
        'file_type': 'json'
    }

    endpoint_state = 'https://api.stlouisfed.org/fred/category/children'
    response_state = requests.get(endpoint_state, params=params_state)
    
    state_counties = {} # instantiate dictionary for MSAs to enure valid return statement
    
    try:
        # "categories call returns both counties and MSAs; we only want MSAs
        county_cat = [s['id'] for s in json.loads(response_state.text)['categories']][0] 

        params_countycat = {
                'api_key' : api_key,
                'category_id' : county_cat,
                'file_type': 'json'
            }

        endpoint_countycat = 'https://api.stlouisfed.org/fred/category/children'

        response_countycat = requests.get(endpoint_countycat, params=params_countycat)

        state_counties = {s['name'] : s['id'] for s in json.loads(response_countycat.text)['categories']}
        
    except:
        pass
    
    return state_counties

def get_loc_series_list(cat, api_key):
    '''
    Get a list of all available FRED series for a particular MSA or county

    Parameters
    ----------
    cat : number cast as string
        FRED category of location
    api_key : string
        API key

    Returns
    -------
    loc_series : dataframe
        Table of all available FRED Series and descriptions for given location

    '''
    
    params = {
    'api_key' : api_key,
    'category_id' : cat,
    'file_type': 'json'
    }
    
    endpoint = 'https://api.stlouisfed.org/fred/category/series'

    response = requests.get(endpoint, params=params)
    
    loc_series = pd.DataFrame()
    
    try:
        loc_series = pd.DataFrame(json.loads(response.text)['seriess'])
        
    except:
        pass
    
    return loc_series

    
def get_series(series_id, api_key):
    '''
    

    Parameters
    ----------
    series_id : string
        the FRED series id
    api_key : string
        FRED API key

    Returns
    -------
    observations : dataframe of observations
        dataframe of observations

    '''
    params = {
        'api_key' : api_key,
        'series_id' : series_id,
        'file_type': 'json'
    }
    
    endpoint = 'https://api.stlouisfed.org/fred/series/observations'
    
    response = requests.get(endpoint, params=params)
    observations = pd.DataFrame(json.loads(response.text)['observations'])
    
    return observations

def get_all_series(api_key, series_df, keyword, msa=True):
    '''

    Parameters
    ----------
    api_key : string
        FRED API KEY.
    series_df : dataframe
        dataframe of FRED MSA series, including columns "title", "id", "state", and "msa"
    keyword : string
        name of overall series (e.g. "Per Capita Personal Income")
    msa : boolean
        whether the location is an msa (True) or county (False)

    Returns
    -------
    output : dataframe
        dataframe with data, value, state, and msa for ALL FRED MSA data series matching the keyword

    '''
    reg_keyword = r'^{}'.format(keyword)
    reg_keyword = reg_keyword.replace('(', '[(]').replace(')', '[)]')
    subseries = series_df[series_df.title.str.match(reg_keyword)==True].reset_index(drop=True)
    
    if msa:
        subseries = subseries.sort_values(['city', 'msa_state', 'frequency', 'seasonal_adjustment'], ascending=[True, True, False, False])
        subseries = subseries.groupby(['city', 'msa_state']).head(1).reset_index()
        
    else:
        subseries = subseries.sort_values(['county', 'state', 'frequency', 'seasonal_adjustment'], ascending=[True, True, False, False])
        subseries = subseries.groupby(['county', 'state']).head(1).reset_index()
        
    output= pd.DataFrame()
    
    for idx in range(len(subseries)):
        series_output = get_series(subseries.id[idx], api_key)
        series_output['msa_state'] = subseries.msa_state[idx]
        series_output['city'] = subseries.city[idx]
        
        if msa == False:
            series_output['county'] = subseries.county[idx]
            series_output['state'] = subseries.state[idx]
            
        series_output['title'] = subseries.title[idx]
        series_output['id'] = subseries.id[idx]
        series_output['frequency'] = subseries.frequency[idx]
        series_output['seasonal_adjustment'] = subseries.seasonal_adjustment[idx]
        
        series_output = series_output.rename(columns={'value' : keyword})
        series_output = series_output.drop(columns=['realtime_start', 'realtime_end'])
        
        output = output.append(series_output)
        time.sleep(3)
        
    return output
    
def save_series_data(api_key, series_df, keyword, msa=True):
    '''
    Parameters
    ----------
    api_key : string
        FRED API KEY.
    series_df : dataframe
        dataframe of FRED MSA series, including columns "title", "id", "state", and "msa"
    keyword : string
        name of overall series (e.g. "Per Capita Personal Income")
    msa : boolean
        whether the location is an msa (True) or county (False)

    Returns
    -------
    saves a csv file with data to fred-data folder
    '''

    df = get_all_series(api_key, series_df, keyword, msa)
    
    if msa:
        filename = '..\\msa-data\\' + keyword.replace(':', ' -') + '.csv'
        
    else:
        filename = '..\\county-data\\' + keyword.replace(':', ' -') + '.csv'
        
    df.to_csv(filename, index=False)
    
    
def save_all_series_data(api_key, series_df, keyword_list, msa=True):
    '''

    Parameters
    ----------
    api_key : string
        FRED API key
    series_df : dataframe
        dataframe of FRED MSA series, including columns "title", "id", "state", and "msa"
    keyword_list : list of strings
        lit of keywords for series
    msa : boolean
        whether the location is an msa (True) or county (False)

    Returns
    -------
    saves csv file with data to fred-data folder for all keywords in keyword list

    '''
    
    for keyword in keyword_list:
        save_series_data(api_key, series_df, keyword, msa)
        
        
######### Section 3 - Combining and Transforming FRED data

def get_new_colnames(filenames):
    '''
    Creates a dictionary of new column names from FRED series names

    Parameters
    ----------
    filenames : set
        set of filenames with relevant data series

    Returns
    -------
    new_col_names : dictionary
        mapping of old names to new names

    '''
    new_col_names = {} # rename columns for ease of use
    for k in filenames:

        if re.match(r'^All Employees', k): # proportion of pop in various industries
            field = str.lower(re.split(' ', re.split('-', k)[1])[1].strip('.csv'))
            new_col_names[k] = field

        elif 'Hourly Earnings' in k:
            new_col_names[k] = 'hourly_earnings'

        elif re.match(r'^Housing Inventory', k): # 
            new_col_names[k] = 'housing_' + str.lower(re.split('- ', k)[1].replace(' ', '_').strip('.csv'))

        else:
            new_col_names[k] = str.lower(k.replace(' ', '_').strip('.csv'))
        
    return new_col_names


def get_employee_cols(col_dict):
    '''
    Extracts column names where series relates to number of employees in a sector

    Parameters
    ----------
    col_dict : dict
        results from get_new_colnames function

    Returns
    -------
    employee_cols : list
        list of col names dealing with employees in sector

    '''
    employee_cols = [v for k, v in col_dict.items() if re.match(r'^All Employees', k)]
    return employee_cols

def import_data_file(file, filepath, col_dict, start_date='1800-01-01', end_date = '2020-01-01'):
    '''
    Imports data and metadata relating to FRED series, saved with specific formats via save_all_series function

    Parameters
    ----------
    file : string
        name of file
        
    filepath : string
        path where files stored
        
    col_dict : dictionary
        crosswalk mapping of FRED series names to col names
        
    start_date : string
        minimum date for pulling data series (INCLUSIVE)

    end_date : string
        maximum date for pulling data series (EXCLUSIVE)

    Returns
    -------
    data_and_meta : dict
        dictionary w dataframe of data and some metadata fields

    '''
    
    data = pd.read_csv(os.path.join(filepath, file))
    frequency = data.frequency[0]
    
    data.date = pd.to_datetime(data.date)
    
    data = data[(data.date >= start_date) & (data.date < end_date)].reset_index(drop=True)
    
    data['year'] = [d.year for d in data.date]
    
    data.columns.values[1] = col_dict[file]
    
    if frequency == 'Monthly':
        data['month'] = [d.month for d in data.date]
        
    if frequency != 'Annual':
        data['quarter'] = [((d.month -1)//3 + 1) for d in data.date]
    
    data = data.dropna() # empirically happens only when series is not yet reported
        
    to_drop = ['title', 'id', 'frequency', 'seasonal_adjustment']
    data_and_meta = {'data' : data.drop(columns=to_drop), 'frequency' : data.frequency[0]}
    
    return data_and_meta


def get_data_dict(file_list, filepath, col_dict, start_date=date(1800, 1, 1)):
    '''
    creates dict of data + metadata for all FRED series in data folder

    Parameters
    ----------
    file_list : set
        set of file names
        
    filepath : string
        path where files stored
        
    col_dict : dictionary
        crosswalk mapping of FRED series names to col names

    start_date : date
        minimum date for pulling data series

    Returns
    -------
    data_dict : dict
        dict of data + metadata for all FRED series in data folder
    
    '''
    
    data_dict = {}

    for file in file_list:
        data_dict[col_dict[file]] = import_data_file(file, filepath, col_dict, start_date)
        
    return data_dict

def convert_dtypes(df, cols_to_ignore):
    '''
    convert columns to float for numeric operations
    
    Parameters
    ----------
    df : dataframe
        any dataframe with columns requiring a float conversion
    cols_to_ignore : list
        a list of columns which should remain their current type

    Returns
    -------
    new_df : dataframe
        updated df with columns converted to float

    '''
    
    new_df = df.copy()
    
    # convert columns to float for numeric operations
    for col in new_df:
        if col not in cols_to_ignore:
            new_df[col] = new_df[col].astype(float)
            
    return new_df

def combine_series(series_dict, base_key, addition_key):
    '''
    combines data results from two different FRED series - useful for when names differ across locations 
    (e.g. Home Price Index vs. House Price Index). Does NOT account for time frequency conversions;
    this must be done beforehand

    Parameters
    ----------
    series_dict : dict
        dictionary of data and metadata from get_data_dict
        
    base_key : string
        key of series to be kept after combination
        
    addition_key : string
        key of series to be deleted after combination

    Returns
    -------
    new_dict : dict
        updated version of data dictionary, with data from both keywords combined into a single series
    
    '''
    new_dict = series_dict.copy()
    
    base_cols = new_dict[base_key]['data'].columns # ensure both data dfs have same columns
    
    addition_df = new_dict[addition_key]['data'].rename(columns={addition_key : base_key}) # rename data column to match base
    
    new_dict[base_key]['data'] = new_dict[base_key]['data'].append(addition_df[base_cols]) # combine onto base key
    
    del new_dict[addition_key] # remove second key
    
    return new_dict

def convert_data_to_df(data_dict, id_vars, target_var):
    '''
    converts data dict from get_data_dict function into df

    Parameters
    ----------
    data_dict : dict
        dictionary of data and metadata from get_data_dict
        
    id_vars : list of columns which contain key variables
        
    target_var : string
        target variable for later analysis

    Returns
    -------
    dataset : df
        df of all variables needed for analysis
    
    '''
    
    # not all id vars in all dfs 
    target_subset = list(set(data_dict[target_var]['data'].columns).intersection(set(id_vars))) + [target_var]
    
    dataset = data_dict[target_var]['data'][target_subset]
    

    for k, v in data_dict.items():
        if k != target_var:
            frequency = v['frequency']
            cols_to_drop = list(set(data_dict[k]['data'].columns).intersection(set(['year', 'quarter']))) # "quarter" not in all 
            series = v['data'].drop(columns=cols_to_drop)
                
            # for monthly, only use start of quarter
            if frequency == 'Monthly':
                series = series.query("month in [1,4,7,10]").drop(columns=['month'])

            dataset = pd.merge(dataset, series,
                               how = 'left', on = ['date', 'city', 'msa_state'], validate = 'one_to_one')

    # interpolate
    annual_cols = [k for k, v in data_dict.items() if v['frequency'] == 'Annual']
    for col in annual_cols:
        dataset[col] = dataset.groupby(['city', 'msa_state'])[col].apply(lambda group: group.interpolate())

    return dataset

def normalize_data(df, cols_to_divide, cols_not_normalized, cols_log_trans):
    
    '''
    performs normalization, conversion to per capita measures, and log transforms

    Parameters
    ----------
    df : df
        df of all vars needed for analysis
        
    cols_to_divide : list of cols to convert to per capita
    
    cols_not_normalized : list of columns to NOT normalize (all rest will be normalized)
    
    cols_log_trans : list of columns to log transform
        
    Returns
    -------
    normalized : df
        df post normalization and prep procedures
    '''
    
    #convert to per capita
    for col in cols_to_divide:
        df[col] = df[col]/df.population
    
    normalized = df.drop(columns = cols_not_normalized).transform(lambda x: (x - x.mean()) / x.std())

    normalized = pd.concat([df[cols_not_normalized], normalized], axis=1)

    for col in cols_log_trans:
        normalized[col] = np.log(normalized[col])
    
    return normalized

def transform_data(df, id_vars, monthly_cols, seasonal_cols, annual_cols, monthly_order, seasonal_order, annual_order):
    
    '''
    converts data to format needed for Variational autoregression

    Parameters
    ----------
    df : df
        df of all vars needed for analysis (post normalization)
        
    monthly_cols : list of cols that report monthly
    
    seasonal_cols : list of cols with seasonality
    
    annual_cols : list of cols that report annually
    
    monthly, seasonal, annual order : number of periods to look back for monthly and annual trends as well as seasonality (12 months previously)
    
        
    Returns
    -------
    transformed : df
        df with current and previous values for variables
    
    '''
    
    output_df = df.copy()

    seasonal = output_df[seasonal_cols]
    monthly = output_df[monthly_cols]
    annual = output_df[annual_cols]

    for i in range(1, monthly_order+1):
        shifted = monthly.shift(i).rename(columns={col : col + '_' + str(monthly_order - i) for col in monthly.columns})
        output_df = pd.concat([output_df, shifted], axis=1)

    for i in range(1, seasonal_order+1):
        shifted = seasonal.shift(i*12).rename(columns={col : col + '_s' + str(seasonal_order-i) for col in seasonal.columns})
        output_df = pd.concat([output_df, shifted], axis=1)

    for i in range(1, annual_order+1):
        shifted = annual.shift(i*12).rename(columns={col : col + '_' + str(annual_order-i) for col in annual.columns})
        output_df = pd.concat([output_df, shifted], axis=1)
        
    return output_df

def time_plot(df, id_vars, cols_to_ignore):
    
    '''
    given a dataframe, key variables, and columns to leave out, plots a grid of time trends
    for each variable by city, with a black dashed line for the start of COVID
    '''
    trim_df = df.drop(columns=cols_to_ignore)
    melted = pd.melt(trim_df, id_vars=id_vars)
    
    g = sns.FacetGrid(melted, col = 'variable', hue='city', col_wrap=3, height=4, aspect=1)
    g.map(plt.plot, 'date', 'value', linewidth=2)
    #g.map(plt.axvline(x='observed'))
    g.set_titles('{col_name}', fontsize=16)
    g.set_axis_labels('Date', 'Normalized/Log Value', fontsize=16)
    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle('Time Trends for Predictor Variables', fontsize=16)
    g.add_legend()

    dates = [date(2020, 3, 11)]*len(melted.variable.unique())

    for ax, pos in zip(g.axes.flat, dates):
        ax.axvline(x=pos, color='black', linestyle='--')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

def median_scatter(df, id_vars, target_var, **cols_to_ignore):
    '''
    given a dataframe, key variables, target vars, and optionally columns to leave out, 
    plots a grid of scatterplots of target vs predictor for each predictor, by city
    '''
    trim_df = df.drop(columns=cols_to_ignore)
    melted = pd.melt(trim_df, id_vars = id_vars + [target_var])

    g = sns.FacetGrid(melted, col = 'variable', hue='city', col_wrap=3, height=4, aspect=1)
    g.map(plt.scatter, 'value', target_var, alpha=0.2)
    #g.map(plt.axvline(x='observed'))
    g.set_titles('{col_name}', fontsize=16)
    g.set_axis_labels('Value', 'Median Housing Price', fontsize=16)
    g.fig.subplots_adjust(top=.9)
    g.fig.suptitle('Scatter for Predictor Variables', fontsize=16)
    g.add_legend()