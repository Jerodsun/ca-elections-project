# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:59:47 2019

@author: User

QGIS Process:
Have 2 tables/shapefiles: census and precincts
Calculate new area for both
Union the two CENSUS to PRECINCT

Union table: calculate new area of union v_area
Set logic: The census one theoretically covers the entire census area 
(or if not no way of telling at this point right?)

Calculate new field: pct_precinct: v_area / p_area 
Percent of the precinct - we will use that to mask over the precinct results.

Then we're done and export.

Name is not distinct. select distinct NAMELSAD, COUNTYFP from census_tract_area;
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Import census demographic data
census = pd.read_csv('census_arcgis_ca.csv', encoding='latin-1')


# Import Premade combined tracts
df = pd.read_csv('state_g18_sr_blk_map.csv')

# drop unmatched (did I do this previously in QGIS?)
df = df.dropna(subset=['SRPREC'])

#pct16_reduc has the county code appended
df['pct_precinct'] = df['v_area']/df['area_p']

df_reduc = df[df['pct_precinct'] > 0.02]

# Import precinct election results, 2016 all
election = pd.read_csv('2018-election-data/all_precinct_results_2016.csv')

election_tmp = election[['pct16', 'pres_clinton', 'pres_trump', 'ussenate_harris', 'ussenate_sanchez']]

election_tmp = election_tmp.iloc[:24568] # hack to remove row summations
# during merging it will go away, but here for visual reference anyway

election_tmp['COUNTY'] = election_tmp['pct16'].str[:3]
election_tmp['COUNTY'] = election_tmp['COUNTY'].astype(int)
election_tmp['PCT'] = election_tmp['pct16'].str[4:]



df['COUNTY'] = df['COUNTY'].astype(int)
df['PCTSRPREC'] = df['PCTSRPREC'].astype(float)

# Merge QGIS with precincts
mergers = df_reduc.merge(election_tmp, on="pct16")

# Take slice only for now; this is duplicable
# Apply pct_precinct scaling for each precinct in census tract
manip = mergers.copy()
manip[['clinton', 'trump', 'harris', 'sanchez']] = mergers[['pres_clinton', 'pres_trump', 'ussenate_harris', 'ussenate_sanchez']].multiply(mergers['pct_precinct'], axis="index")

# GROUPBY to combine
test = manip.groupby(['OBJECTID'])[['clinton', 'trump', 'harris', 'sanchez', 'v_area']].sum().reset_index()

# Now merge with census data.

output = test.merge(census, on=['OBJECTID'])

output['pct_hispanic'] = output['HISPANIC']/output['POP2010']
output['pct_sanchez'] = output['sanchez']/(output['harris'] + output['sanchez'])
output['pct_trump'] = output['trump']/(output['trump'] + output['clinton'])

output_sig = output[(output['clinton'] > 10) & (output['trump'] > 10)]

# Btw I think these are cur estimates
#output[output['HISPANIC'] > output['POPULATION']]
# earlier: wtf why is % above 1

# Visualizations
plt.title('2016 Election Exploration')
plt.scatter(output['HISPANIC']/output['POPULATION'], output['clinton']/output['trump'])
plt.scatter(output['HISPANIC']/output['POPULATION'], output['sanchez']/output['harris'])
plt.legend(["Clinton/Trump", "Sanchez/Harris"])
plt.xlabel('Percent Hispanic')
plt.ylabel('Vote Ratio')


plt.title('2016 Election Exploration')
plt.scatter(output_sig['pct_hispanic'], output_sig['pct_sanchez'])
plt.legend()
plt.xlabel('Percent Hispanic')
plt.ylabel('Percent vote for Sanchez')


mapping = pd.read_csv('county_code_map.csv')
mapping = mapping.set_index('FIPS_code')

ls = list(output['CNTY_FIPS'].value_counts().index[:10])
plt.title('2016 Election Exploration')
plt.xlabel('Percent Hispanic')
plt.ylabel('Percent vote for Trump')
for i in ls:
    tmp = output_sig[output_sig['CNTY_FIPS'] == i]
    plt.scatter(tmp['pct_hispanic'], tmp['pct_trump'])
plt.legend([mapping.loc[i]['county'] for i in ls])

    
sd = output[output['CNTY_FIPS'] == 73]

plt.title('2016 Election Exploration')
plt.scatter(sd['pct_hispanic'], sd['pct_trump'])
plt.xlabel('Percent Hispanic')
plt.ylabel('Percent vote for Trump')

output['pct_non_h_white'] = ( output['WHITE'] - output['HISPANIC'] ) /output['POP2010']


plt.title('2016 Election Exploration')
plt.scatter(output['pct_non_h_white'], output['pct_trump'])
plt.xlabel('Percent Non Hispanic White')
plt.ylabel('Percent vote for Trump')

output[['clinton', 'trump', 'harris', 'sanchez']].sum() # slippage count

#8916878

ls = list(output['CNTY_FIPS'].value_counts().index[:10])
plt.title('2016 Election Exploration')
plt.xlabel('Percent Non Hispanic White')
plt.ylabel('Percent vote for Trump')
for i in ls:
    tmp = output_sig[output_sig['CNTY_FIPS'] == i]
    plt.scatter(tmp['pct_non_h_white'], tmp['pct_trump'])
plt.legend([mapping.loc[i]['county'] for i in ls])

plt.title('2016 Election Exploration')
plt.scatter(output['POP10_SQMI'], output['pct_trump'])
plt.xlabel('Population Density / Square Mile')
plt.ylabel('Percent vote for Trump')



#pd.set_option('display.float_format', '{:.2f}'.format)
#Parse results:
#clinton   8916878.19
#trump     4576116.92
#harris    7649879.62
#sanchez   4843210.37

#Actual:
#clinton    8753788
#trump      4483810
#harris     7542753
#sanchez    4701417

# Total slippage:
# (8916878 - 8753788) / 8753788 = 0.0186307916070163
# (4576116 - 4483810) / 4483810 = 0.020586510133123393

# Graphing exploration

# Instead of ratio, next time use % of total vote

