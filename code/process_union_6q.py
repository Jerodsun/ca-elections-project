# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:48:31 2019

@author: Jerod

This is the Q Process: Files from ArcGIS -> census_arcgis_ca.csv

QGIS Process:
Have 2 tables/shapefiles: census and precincts
Calculate new area for both
Union CENSUS to PRECINCT

Union table: calculate new area of union v_area
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Import census demographic data - this is the arcgis version
census = pd.read_csv('census_arcgis_ca.csv', encoding='latin-1')

# Import QGIS combined block group
df = pd.read_csv('census_precinct_min.csv')

#calculate coverage % of each row over the whole census area
df['pct_whole'] = df['area_v']/df['area_p']

#filter out null values - 2 % is arbitrary, negative offset
df_reduc = df[df['pct_whole'] > 0.02]

# Import precinct election results, 2016 all
election = pd.read_csv('all_precinct_results_2016.csv')

# Working subset
election_tmp = election[['pct16', 'pres_clinton', 'pres_trump', 'ussenate_harris', 'ussenate_sanchez']]
election_tmp = election_tmp.iloc[:24568] # remove precinct summations for now

# Not sure if still need this?
# election_tmp['state_ic'] = election_tmp['pct16'].str[:3]
# election_tmp['state_ic'] = election_tmp['state_ic'].astype(int)
# election_tmp['PCT'] = election_tmp['pct16'].str[4:]

# Merge QGIS with precincts
cen_to_precinct = df_reduc.merge(election_tmp, on="pct16")

# Mergers is one-to-many; QGIS combined block group is from census to precinct.

# Take slice only for now; this is duplicable
# Apply pct_whole scaling for each slice in census block group
# This creates new columns reflecting the census block group (OBJECTID) percent of the
manip = cen_to_precinct.copy()
manip[['clinton', 'trump', 'harris', 'sanchez']] = cen_to_precinct[['pres_clinton', 'pres_trump', 'ussenate_harris', 'ussenate_sanchez']].multiply(cen_to_precinct['pct_whole'], axis="index")

# GROUPBY census block group and combine
test = manip.groupby(['OBJECTID'])[['clinton', 'trump', 'harris', 'sanchez', 'v_area']].sum().reset_index()

# len(test) # 22827
# len(census) # 23194 Throughout the QGIS process this is the census block group count that failed to match

# Now merge with census data.
output = test.merge(census, on=['OBJECTID'])

output['pct_hispanic'] = output['HISPANIC']/output['POP2010']
output['pct_sanchez'] = output['sanchez']/(output['harris'] + output['sanchez'])
output['pct_trump'] = output['trump']/(output['trump'] + output['clinton'])
output['pct_non_h_white'] = ( output['WHITE'] - output['HISPANIC'] ) /output['POP2010']

output_sig = output[(output['clinton'] > 10) & (output['trump'] > 10)]

# Btw I think these are cur estimates
#output[output['HISPANIC'] > output['POPULATION']]
# earlier: wtf why is % above 1

# Visualizations
#plt.title('2016 Election Exploration')
#plt.scatter(output['HISPANIC']/output['POPULATION'], output['clinton']/output['trump'])
#plt.scatter(output['HISPANIC']/output['POPULATION'], output['sanchez']/output['harris'])
#plt.legend(["Clinton/Trump", "Sanchez/Harris"])
#plt.xlabel('Percent Hispanic')
#plt.ylabel('Vote Ratio')

#
#plt.title('2016 Election Exploration')
#plt.scatter(output_sig['pct_hispanic'], output_sig['pct_sanchez'])
#plt.legend()
#plt.xlabel('Percent Hispanic')
#plt.ylabel('Percent vote for Sanchez')


mapping = pd.read_csv('county_code_map.csv')
mapping = mapping.set_index('FIPS_code')

ls = list(output['CNTY_FIPS'].value_counts().index[:10])
#plt.title('2016 Election Exploration')
#plt.xlabel('Percent Hispanic')
#plt.ylabel('Percent vote for Trump')
#for i in ls:
#    tmp = output_sig[output_sig['CNTY_FIPS'] == i]
#    plt.scatter(tmp['pct_hispanic'], tmp['pct_trump'])
#plt.legend([mapping.loc[i]['county'] for i in ls])

    
sd = output[output['CNTY_FIPS'] == 73]

#plt.title('2016 Election Exploration')
#plt.scatter(sd['pct_hispanic'], sd['pct_trump'])
#plt.xlabel('Percent Hispanic')
#plt.ylabel('Percent vote for Trump')
#
#
#
#plt.title('2016 Election Exploration')
#plt.scatter(output['pct_non_h_white'], output['pct_trump'])
#plt.xlabel('Percent Non Hispanic White')
#plt.ylabel('Percent vote for Trump')

output[['clinton', 'trump', 'harris', 'sanchez']].sum() # slippage count

#8916878

ls = list(output['CNTY_FIPS'].value_counts().index[:10])

#plt.title('2016 Election Exploration')
#plt.xlabel('Percent Non Hispanic White')
#plt.ylabel('Percent vote for Trump')
#for i in ls:
#    tmp = output_sig[output_sig['CNTY_FIPS'] == i]
#    plt.scatter(tmp['pct_non_h_white'], tmp['pct_trump'])
#plt.legend([mapping.loc[i]['county'] for i in ls])



#plt.title('2016 Election Exploration')
#plt.scatter(output['POP10_SQMI'], output['pct_trump'])
#plt.xlabel('Population Density / Square Mile')
#plt.ylabel('Percent vote for Trump')



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

output_g = output[output['pct_hispanic'] > 0.05]
output_g = output_g.dropna()
X = np.array(output_g['pct_hispanic']).reshape(-1,1)
y = np.array(output_g['pct_sanchez'])
# Election 2018 analysis preprocessing in initial_exp

from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split #, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

alpha = 0.95

clf = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                n_estimators=64, max_depth=3,
                                learning_rate=.08, min_samples_leaf=6,
                                min_samples_split=6)


#{'criterion': 'friedman_mse',
# 'learning_rate': 0.05468716748805014,
# 'max_depth': 1,
# 'max_features': 'auto',
# 'min_samples_leaf': 4,
# 'min_samples_split': 6,
# 'n_estimators': 64}

clf.fit(X, y)

# Make the prediction on the meshed x-axis
y_upper = clf.predict(X)

clf.set_params(alpha=1.0 - alpha)
clf.fit(X, y)

# Make the prediction on the meshed x-axis
y_lower = clf.predict(X)

clf.set_params(loss='ls')
clf.fit(X, y)

# Make the prediction on the meshed x-axis
y_pred = clf.predict(X)

# Plot the function, the prediction and the 90% confidence interval based on
# the MSE
fig = plt.figure()
#plt.plot(X, f(xx), 'g:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
plt.plot(X, y_pred, 'r-', label=u'Prediction')
plt.plot(X, y_upper, 'k-')
plt.plot(X, y_lower, 'k-')
plt.fill(np.concatenate([X, X[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.1, fc='b', ec='None', label='90% prediction interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
#plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.show()



plt.plot(X, y, 'b.')
plt.plot(X, y_upper, 'k.')
plt.plot(X, y_lower, 'k.')
plt.plot(X, y_pred, 'r.', label=u'Prediction')

X_orig = np.array(output_g['pct_hispanic'])

plt.fill_between(X_orig, y_upper, y_lower)

np.concatenate([X, X[::-1]]),
         np.concatenate([y_upper, y_lower[::-1]]),
         alpha=.1, fc='b', ec='None', label='90% prediction interval')