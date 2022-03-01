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

##########
# STEP 1 - PULL DATA FROM CSVS
##########

msa_filepath = '..\\msa-data'
msa_files = os.listdir(msa_filepath) # all msa files

county_filepath = '..\\county-data'
county_files = os.listdir(county_filepath) # all msa files


# remove python checkpoints folders
try:
    msa_files.remove('.ipynb_checkpoints')
except ValueError:
    pass

try:
    county_files.remove('.ipynb_checkpoints')
except ValueError:
    pass

 # dict to rename columns from FRED series names
file_to_col = {
    'All-Transactions House Price Index.csv' : 'hpi',
    'Average Weekly Wages for Employees in Total Covered Establishments.csv' : 'weekly_wages',
    'Home Price Index (Middle Tier).csv' : 'hpi-large',
    'Housing Inventory - Median Listing Price.csv' : 'median_listing_price',
    'Regional Price Parities.csv' : 'rpp',
    'Resident Population.csv' : 'population',
    'Total Gross Domestic Product.csv' : 'gdp',
    'Unemployment Rate.csv': 'unemployment_rate',
    'Combined Violent and Property Crime Offenses.csv' : 'crimes',
    'Equifax Subprime Credit Population.csv' : 'subprime',
    'Estimated Percent of People of All Ages in Poverty.csv' : 'poverty',
    'New Patent Assignments.csv' : 'patents',
    'Number of Private Establishments for All Industries.csv' : 'private_establishments',
    'Premature Death Rate.csv' : 'premature_death_rate',
    'SNAP Benefits Recipients.csv' : 'snap'
}


# bring in msa data as a dict and change dtypes

msa_data_dict = fred_msa.get_data_dict(msa_files, msa_filepath, file_to_col, '2008-01-01') # pull in data and metadata

msa_id_vars = ['date', 'year', 'month', 'city', 'msa_state', 'quarter'] # id variables - not predictors

for k, v in msa_data_dict.items():
    v['data'] = fred_msa.convert_dtypes(v['data'], msa_id_vars)

#data = fred_msa.convert_data_to_df(data_dict, ['date', 'year', 'month', 'city', 'state'], 'housing_median_listing_price') # convert to df

county_data_dict = fred_msa.get_data_dict(county_files, county_filepath, file_to_col, '2008-01-01') # pull in data and metadata

county_id_vars = ['date', 'year', 'quarter', 'month', 'city', 'county', 'state', 'msa_state'] # id variables - not predictors

for k, v in county_data_dict.items():
    v['data'][k] = v['data'][k].replace('.', np.nan) # some nulls designated with "."
    v['data'] = fred_msa.convert_dtypes(v['data'], county_id_vars)
    
    # interpolate missing values
    v['data'][k] = v['data'].groupby(['city', 'state', 'county'])[k].apply(lambda group: group.interpolate(method='linear', limit=20, limit_direction='both'))

##########
# STEP 2 - COMBINE HPI FOR SMALL AND LARGE CITIES
##########

# convert monthly HPI for large cities to quarterly
msa_data_dict['hpi-large']['data'] = msa_data_dict['hpi-large']['data'].query("month in [1,4,7,10]")

# combine hpi for small and large cities
msa_data_dict = fred_msa.combine_series(msa_data_dict, 'hpi', 'hpi-large')

##########
# STEP 3 - LINK HPI TO MEDIAN HOUSING PRICE
##########

# get median listing price at start of 2020
med_prices_2020 = msa_data_dict['median_listing_price']['data'].query(
    "date=='2020-01-01'").drop(columns=['date', 'year', 'month', 'quarter']).reset_index(drop=True)

# remove listing price from dictionary (no longer needed)
del msa_data_dict['median_listing_price']

# incorporate LA
la_med_listing_id = 'MEDLISPRI6037'

api_key = 'a37b50cd27afbc3ce23a81ddc5541dec'

la_2020_med_price = float(fred_msa.get_series(la_med_listing_id, api_key).query("date=='2020-01-01'").value)

med_prices_2020 = med_prices_2020.append({'city' : 'Los Angeles',
                       'msa_state' : 'CA',
                       'median_listing_price' : la_2020_med_price}, ignore_index=True)

# create HPI multiplier to compare between cities, based on median in Q1 2020
med_prices_2020['multiplier'] = med_prices_2020.median_listing_price/np.median(med_prices_2020.median_listing_price)

# use merge to add multiplier to HPI dataset
msa_data_dict['hpi']['data'] = pd.merge(
    msa_data_dict['hpi']['data'], med_prices_2020.drop(
    columns=['median_listing_price']), how='left', on=['city', 'msa_state'], validate='many_to_one')

# calculate adjusted HPI
msa_data_dict['hpi']['data']['hpi'] = msa_data_dict['hpi']['data'].hpi*msa_data_dict['hpi']['data'].multiplier

##########
# STEP 4 - AGGREGATING COUNTY DATA
##########

# extract and remove population from other data series

county_pop = county_data_dict['population']['data']
del county_data_dict['population']

# sum county populations across MSAs
msa_pops = pd.DataFrame(county_pop.groupby(['year', 'msa_state', 
                                            'city'])['population'].sum()).reset_index().rename(columns={'population' : 'msa_pop'})

county_pop = pd.merge(county_pop, msa_pops, how = 'left', on=['year', 'msa_state', 'city'], validate='many_to_one')

# calculate weights for each county
county_pop['pop_fraction'] = county_pop.population/county_pop.msa_pop

# add population column to each series
for k, v in county_data_dict.items():
    v['data'] = pd.merge(v['data'], county_pop.drop(columns=['date', 'msa_state', 'city']),
                        how='left', on=['county', 'state', 'year'])

weighted_avg_series = ['subprime', 'poverty', 'premature_death_rate']

# aggreagte - either by sum or weighted average
for k, v in county_data_dict.items():
    
    if k in weighted_avg_series:
        v['data'][k] = v['data'][k]*v['data']['pop_fraction']
    
    if v['frequency'] == 'Annual':
        grouped = v['data'].groupby(['date', 'city', 'msa_state', 'year'])

    elif v['frequency'] == 'Quarterly':
        grouped = v['data'].groupby(['date', 'city', 'msa_state', 'year', 'quarter'])

    else:
        grouped = v['data'].groupby(['date', 'city', 'msa_state', 'year', 'quarter', 'month'])
        
        
    aggregated = pd.DataFrame(grouped[k].sum()).reset_index()
        
    v['data'] = aggregated

##########
# STEP 5 - CREATE AND SAVE UNIFIED DF
##########

combined_data_dict = msa_data_dict | county_data_dict

hpi_data = fred_msa.convert_data_to_df(combined_data_dict, msa_id_vars, 'hpi')

hpi_data.to_csv('..\\working-data\\hpi-data.csv', index=False)