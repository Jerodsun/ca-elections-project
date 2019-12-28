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
census = pd.read_csv('nhgis0003_modified.csv', encoding='latin-1')
census_ca = census[census['STATE'] == 'California']

# Import QGIS combined tracts
df = pd.read_csv('union_calc_ca_all_reduc.csv')
#pct16_reduc has the county code appended
df['pct_precinct'] = df['area_v']/df['area_p']


# dedupe exploration
dedupe = df.drop_duplicates(['COUNTYFP', 'NAMELSAD'])
# sort by layer; overlap matches from other counties... can be discarded

#df = df.dropna(subset=['layer']) # no corresponding: there are 86489 - 83648
# better yet use area_v
df = df.dropna(subset=['area_v']) # 86489 - 77227 = 9262

df['matcher'] = df['layer'].str.replace('\-(.*)', '')
df['matcher'] = df['matcher'].astype(int)
df['COUNTYFP'] = df['COUNTYFP'].astype(int)


df[df['matcher'] == df['COUNTYFP']] #77227 - 75055 = 2172 wrong county
mismatch = df[df['matcher'] != df['COUNTYFP']]
dedupe = df.drop_duplicates(['COUNTYFP', 'NAMELSAD']) # Census duplicates

df_reduc = df[df['pct_precinct'] > 0.02]

# Import precinct election results, 2016 all
election = pd.read_csv('2018-election-data/all_precinct_results_2016.csv')

election_tmp = election[['pct16', 'pres_clinton', 'pres_trump', 'ussenate_harris', 'ussenate_sanchez']]

# Merge QGIS with precincts
mergers = df_reduc.merge(election_tmp, on="pct16")

# Take slice only for now; this is duplicable
# Apply pct_precinct scaling for each precinct in census tract
manip = mergers.copy()
manip[['clinton', 'trump', 'harris', 'sanchez']] = mergers[['pres_clinton', 'pres_trump', 'ussenate_harris', 'ussenate_sanchez']].multiply(mergers['pct_precinct'], axis="index")

# GROUPBY to combine
test = manip.groupby(['NAME', 'NAMELSAD'])[['clinton', 'trump', 'harris', 'sanchez', 'area_v']].sum().reset_index()

# Area confirmation to see slippage
census_area = manip[['NAME', 'NAMELSAD', 'area_c', 'COUNTYFP']].drop_duplicates(['NAME'])
fin = test.merge(census_area, on=["NAME", "NAMELSAD"])

# Then groupby all the others then sum.

# Area-weighted. Now merge with census data.

census_ca['NAME'] = census_ca['NAME_E'].str.replace('\,(.*)', '')
output = fin.merge(census_ca, left_on=['NAMELSAD', 'COUNTYFP'], right_on=['NAME', 'COUNTYA'])

# Visualizations
plt.title('2016 Election Exploration')
plt.scatter(output['AHZAE012']/output['AHZAE001'], output['clinton']/output['trump'])
plt.scatter(output['AHZAE012']/output['AHZAE001'], output['sanchez']/output['harris'])
plt.legend(["Clinton/Trump", "Sanchez/Harris"])
plt.xlabel('Percent Hispanic')
plt.ylabel('Vote Ratio')
    
plt.title('2016 Election Exploration')
plt.scatter(output['AHZAE012']/output['AHZAE001'], output['sanchez']/output['harris'])
plt.legend(["Sanchez/Harris"])
plt.xlabel('Percent Hispanic')
plt.ylabel('Vote Ratio')
    


output[['clinton', 'trump', 'harris', 'sanchez']].sum() # slippage count

#pd.set_option('display.float_format', '{:.2f}'.format)
#Parse results:
#clinton   8358011.25
#trump     4355114.25
#harris    7201545.97
#sanchez   4553547.09

#Actual:
#clinton    735746
#trump      477766
#harris     7542753
#sanchez    4701417

# Total slippage:
# (8753788 - 8358011) / 8753788 = 0.04521208418
# (4483810 - 4355114) / 4483810 = 0.02870237588



# Graphing exploration

# Instead of ratio, next time use % of total vote

from sklearn.linear_model import LinearRegression
model = LinearRegression()

p = pd.concat([output['AHZAE012']/output['AHZAE001'], output['sanchez']/output['harris']], axis=1)
p = p.dropna()
model.fit(np.array(p[0]).reshape(-1,1), p[1])
x_new = np.linspace(0, 1, 100)
y_new = model.predict(x_new[:, np.newaxis])

ax = plt.axes()
ax.set_ylabel('Sanchez/Harris')
ax.set_xlabel('Percent Hispanic')
ax.set_title('2016 Senate Election Vote Characteristics')
ax.scatter(p[0], p[1])
ax.plot(x_new, y_new, 'm')

x_score = np.linspace(0, 1, 101)
y_score = model.predict(x_score.reshape(-1,1))

# stick with this for now

from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split #, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
regressor = GradientBoostingRegressor(verbose = 1)

param_dist = {"max_depth": randint(1, 11),
              "learning_rate": uniform(),
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": randint(2, 50),
              "min_samples_leaf": randint(1, 11),
              "n_estimators": randint(50,150),
              "criterion": ["friedman_mse"]}



X_train, X_test, y_train, y_test = train_test_split(p[0], p[1], test_size=0.25, random_state=5150)



#random_search.best_params_
#{'criterion': 'friedman_mse',
# 'learning_rate': 0.05468716748805014,
# 'max_depth': 1,
# 'max_features': 'auto',
# 'min_samples_leaf': 4,
# 'min_samples_split': 6,
# 'n_estimators': 64}

n_iter_search = 100

def exp_mean_squared_error(estimator, X, y):
    y_predicted = np.exp(estimator.predict(X))
    y_true = np.exp(y)
    return -np.sqrt(mean_squared_error(y_true, y_predicted))

random_search = RandomizedSearchCV(regressor, 
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   scoring = exp_mean_squared_error,
                                   random_state = 5150)

# The ML process
random_search.fit(np.array(p[0]).reshape(-1,1), p[1])


random_search.fit(np.array(X_train).reshape(-1,1), y_train)


plt.plot(x_score, y_score, 'r')
plt.scatter(p[0], p[1])


x_score = np.linspace(0, 1, 201)
y_score = random_search.best_estimator_.predict(x_score.reshape(-1,1))
plt.plot(x_score, y_score, 'r')
plt.scatter(p[0], p[1])


default_regressor= regressor.fit(np.array(p[0]).reshape(-1,1), p[1])
y_score = default_regressor.predict(x_score.reshape(-1,1))
plt.plot(x_score, y_score, 'r')
plt.scatter(p[0], p[1])