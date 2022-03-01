##########
# STEP 1 - IMPORT PACKAGES AND METADATA
##########

# import packages

import pandas as pd
import numpy as np
import requests
import json
import os
import time
import fred_msa
import regex as re

# import datasets

data_path = '..\\raw-data'

# city-county crosswalk
crosswalk = pd.read_csv(os.path.join(data_path, 'msa-county-crosswalk.csv'), encoding='cp1252', 
                        usecols=['code', 'county', 'msa'])

# remove nulls
crosswalk = crosswalk[crosswalk.msa.notna()]
crosswalk['city'] = crosswalk.msa.apply(fred_msa.split_city)
crosswalk['state'] = crosswalk.msa.apply(fred_msa.extract_state)

# msas used in analysis
msas_to_use = pd.read_csv(os.path.join(data_path, 'msas-to-use.csv'))

# state abbreviations
states = pd.read_csv(os.path.join(data_path, 'states.csv'))

# dictionaries for state abbreviations (needed b/c states written out in county crosswalk but
# abbreviated in msa and county id dfs)
state_to_ab = {states.loc[idx, 'State'] : states.loc[idx, 'Code'] for idx in range(len(states))}
ab_to_state = {states.loc[idx, 'Code'] : states.loc[idx, 'State'] for idx in range(len(states))}

# get list of relevant counties (counties in relevant msas)

counties_to_use = pd.merge(msas_to_use, crosswalk, how='left', on=['city', 'state'], validate='one_to_many')

counties_to_use = counties_to_use.rename(columns={'city' : 'msa_city', 'state' : 'msa_state'})

##########
# STEP 2 - SET UP PARAMETERS FOR API
##########

# extract state IDs for FRED API

api_key = 'a37b50cd27afbc3ce23a81ddc5541dec'
endpoint = 'https://api.stlouisfed.org/fred/series/categories'

state_ids = fred_msa.get_state_ids(api_key)

##########
# STEP 3 - GET ALL MSA AND COUNTY IDS
##########

msa_ids = pd.DataFrame() # empty df for msa data
county_ids = pd.DataFrame() # empty df for county data

for state, state_id in state_ids.items(): # loop through states
    state_msa_dict = fred_msa.get_msa_cats(api_key, state_id) # get state msa data from API
    
    state_msa_dict = {k : v for k, v in state_msa_dict.items() if '(CMSA)' not in k}
    
    if len(state_msa_dict) > 0: # some states have no MSAs
        
        states_msa = pd.Series(state_msa_dict.keys()).apply(fred_msa.extract_state) # extract state name from msa
        
        # create df with metadata for state's msas
        state_msa_df = pd.DataFrame({'city' : pd.Series(list(state_msa_dict.keys())).apply(fred_msa.split_city),
                                     'msa_state' : states_msa, 'ID' : list(state_msa_dict.values())})
        
        state_msa_df['state'] = state_to_ab[state]
        
        msa_ids = msa_ids.append(state_msa_df) # append to larger df with msa info for all states
    
    time.sleep(3) # ensure process not shut out for too many API calls
    
    state_county_dict = fred_msa.get_county_cats(api_key, state_id) # get state county data from API
    states_county = pd.Series(state_county_dict.keys()).apply(fred_msa.extract_state) # extract state name 
    
    # df with state county info
    state_county_df = pd.DataFrame({'county' : list(state_county_dict.keys()),
                                    'state' : states_county, 'ID' : list(state_county_dict.values())})
    
    county_ids = county_ids.append(state_county_df) # append to larger df with county info for all states
    
    time.sleep(3) # ensure process not shut out for too many API calls


##########
# STEP 4 - CLEAN UP DATFRAMES AND MERGE
##########

# some msas may have duplicates
msa_ids = msa_ids.drop_duplicates(subset=['city', 'state', 'msa_state'])

# most msas span multiple states and will therefore be duplicated - match to home state
msa_ids = msa_ids[msa_ids.msa_state == msa_ids.state]

# convert to abbreviation in counties df
counties_to_use['county_state'] = counties_to_use.county.apply(fred_msa.extract_state).map(state_to_ab)

# remove state from county name
county_ids.county = [re.split(',', c)[0] for c in county_ids.county]
counties_to_use.county = [re.split(',', c)[0] for c in counties_to_use.county]

# merge county crosswalk with county ids (get county ids for relevant counties only)
counties_to_use = pd.merge(counties_to_use, county_ids, how='left', left_on=['county', 'county_state'],
         right_on=['county', 'state'], validate='one_to_one')

# get msa ids and metadata for only relevant msas
msas_to_use = pd.merge(msas_to_use, msa_ids,
         how='left', on=['city', 'state'], validate='one_to_one')

##########
# STEP 5 - GET LIST OF AVAILABLE SERIES FOR EACH MSA
##########

all_msa_series = pd.DataFrame()

for idx in range(len(msas_to_use)):
    msa_series = fred_msa.get_loc_series_list(msas_to_use.ID[idx], api_key)
    msa_series['city'] = msas_to_use.city[idx]
    msa_series['msa_state'] = msas_to_use.state[idx]
    
    all_msa_series = all_msa_series.append(msa_series).reset_index(drop=True)
    time.sleep(3)

all_msa_series.to_csv('..\\metadata\msa-series.csv', index=False)

##########
# STEP 6 - GET LIST OF AVAILABLE SERIES FOR EACH COUNTY
##########

all_county_series = pd.DataFrame()

for idx in range(len(counties_to_use)):
    county_series = fred_msa.get_loc_series_list(counties_to_use.ID[idx], api_key)
    county_series['county'] = counties_to_use.county[idx]
    county_series['state'] = counties_to_use.state[idx]
    county_series['city'] = counties_to_use.msa_city[idx]
    county_series['msa_state'] = counties_to_use.msa_state[idx]
    
    all_county_series = all_county_series.append(county_series).reset_index(drop=True)
    time.sleep(3)

all_county_series.to_csv('..\\metadata\county-series.csv', index=False)