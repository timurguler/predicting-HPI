##########
# This script pulls data for house price analysis for all MSAs included in the analysis, 
# using the FRED API to do this at the MSA and county level. This requires two table with 
# all available series for these locations - one at the MSA level and one at the county level. 
# This will have been created as a result of script 1, "1-get-mas-and-county-series-list". 
# Keywords must be exact, unique partial matches and must contain the beginning of the series name.
##########

import pandas as pd
import numpy as np
import os
import requests
import json
import time
import matplotlib.pyplot as plt
import fred_msa

api_key = 'a37b50cd27afbc3ce23a81ddc5541dec'

msa_series = pd.read_csv('..\\metadata\\msa-series.csv')
county_series = pd.read_csv('..\\metadata\\county-series.csv')

msa_keywords = [
    'Home Price Index (Middle Tier)',
    'All-Transactions House Price Index',
    'Housing Inventory: Median Listing Price',
    'Unemployment Rate',
    'Total Gross Domestic Product',
    'Resident Population',
    'Regional Price Parities',
    'Average Weekly Wages for Employees in Total Covered Establishments',     
               ]

county_keywords = [
    'SNAP Benefits Recipients',
    'Estimated Percent of People of All Ages in Poverty',
    'Number of Private Establishments for All Industries',
    'New Patent Assignments',
    'Premature Death Rate',
    'Equifax Subprime Credit Population',
    'Combined Violent and Property Crime Offenses' 
    'Resident Population'
]

county_keywords = ['Resident Population']

#fred_msa.save_all_series_data(api_key, msa_series, msa_keywords)
fred_msa.save_all_series_data(api_key, county_series, county_keywords, msa=False)