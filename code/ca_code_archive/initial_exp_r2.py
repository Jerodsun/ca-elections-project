# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:24:54 2019

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#df = pd.read_csv('precinct_2018.csv', encoding="ansi")
#2.4 GB of memory

#df['state'].value_counts()
#ca_only = df[df['state'] == "California"]
#ca_only["office"].value_counts()

df = pd.read_csv('2018-election-data/precinct_2018_ca.csv')

df['office'].value_counts()

np.random.seed(0)

#
#ca_gov = df[df['office'] == "Governor"]
#ca_insur = df[df['office'] == "Insurance Commissioner"]
#ca_prop_6 = df[df['office'] == "Proposition 6"]
#ca_lt_gov = df[df['office'] == "Lieutenant Governor"]


# Basic process
# =============================================================================
# reduc = df[["precinct", "party", "candidate", "votes", "county", "office"]]
# rreduc= reduc[reduc["office"].isin(["Governor"])]
# result = rreduc.set_index(['office', 'precinct', 'county', 'candidate'])['votes'].unstack()
# result = result.reset_index()
# =============================================================================

# Propositions are weird - party is nan, office still there.

def reduce(df_slice):
    """ Calls unstack() to combine rows into columns """
    reduc = df_slice[["precinct", "party", "candidate", "votes", "county", "office"]]
    result = reduc.set_index(['office', 'precinct', 'county', 'candidate'])['votes'].unstack()
    return result.reset_index()

init = reduce(df[df['office'] == "Governor"])
#del init['office']





# avoid pandas visualization overload
for i in ("Lieutenant Governor", "Insurance Commissioner", "Proposition 6", "State Treasurer"):
    df_reduc = reduce(df[df['office'] == i])
#    del df_reduc['office']
    init = init.merge(df_reduc, on=["precinct", "county"])


sd_vote = init[init['county'] == "San Diego"]

ax = sd_vote.plot(x="Gavin Newsom", y="Eleni Kounalakis",kind="scatter", color="b")
sd_vote.plot(x="Gavin Newsom", y="Ed Hernandez", ax=ax, kind="scatter", color="r", legend=True)


#ax = mergg.plot(x = 'total_vote_gov', y = "vote_decr", kind = "scatter", color = 'DarkBlue', label="Gov")
#mergg.plot(x="total_vote_insur", y = "vote_decr", kind="scatter", ax=ax, label = "Insur", color = "DarkGreen")

ca_lt_gov_reduc = reduce(df[df['office'] == "Insurance Commissioner"])


#ca_gov_reduc_sd = ca_gov_reduc[ca_gov_reduc["county"] == "San Diego"]
#ca_insur_reduc_sd = ca_insur_reduc[ca_insur_reduc["county"] == "San Diego"]

plt.title("Results by Precinct: 2018 California Statewide Election")
plt.plot(init['Gavin Newsom'], init['John H. Cox'], 'b.')
plt.plot(init['Ricardo Lara'], init['Steve Poizner'], 'r.')
plt.legend(['Governor', 'Insurance Commissioner'])
plt.ylabel('Republican / Independent')
plt.xlabel('Democrat')

# Propensity of each individual district to be "swing voters"
# By precinct TYPE (if it exists, idk if there's such data)

plt.title("Results by Precinct: 2018 California Statewide Election")
plt.plot(ca_gov_reduc_sd['democrat'], ca_gov_reduc_sd['republican'], 'b.')
plt.plot(ca_insur_reduc_sd['democrat'], ca_insur_reduc_sd['independent'], 'r.')
plt.legend(['Governor', 'Insurance Commissioner'])
plt.ylabel('Republican / Independent')
plt.xlabel('Democrat')

# Get general data first then precinct by precinct data

# Segregate by those precincts most likely to contain "Swing voters"

# For either political party Split ticket voters / single issue voters / etc.

mergg = ca_gov_reduc_sd.merge(ca_insur_reduc_sd, on="precinct")
mergg['democrat_x'].sum() # 658346, same as Wikipedia check

#mergg = mergg[(mergg["independent"] > 1) & (mergg["democrat_x"] > 1)]
mergg['total_vote_gov'] = mergg['democrat_x'] + mergg['republican']
mergg['total_vote_insur'] = mergg['democrat_y'] + mergg['independent']
mergg['vote_decr'] = mergg['total_vote_gov'] - mergg['total_vote_insur']

mergg['delta_d'] = mergg['democrat_x'] - mergg['democrat_y']
mergg['delta_r'] = mergg['republican'] - mergg['independent']

plt.title("Results by Precinct: 2018 California Statewide Election")
plt.plot(mergg['democrat'], ca_gov_reduc_sd['republican'], 'b.')
plt.plot(ca_insur_reduc_sd['democrat'], ca_insur_reduc_sd['independent'], 'r.')
plt.legend(['Governor', 'Insurance Commissioner'])
plt.ylabel('Republican / Independent')
plt.xlabel('Democrat')


# random stuff testing
ax = mergg.plot(x = 'total_vote_gov', y = "vote_decr", kind = "scatter", color = 'DarkBlue', label="Gov")
mergg.plot(x="total_vote_insur", y = "vote_decr", kind="scatter", ax=ax, label = "Insur", color = "DarkGreen")
# To plot multiple column groups, repeat plot specifying target ax - pandas visualization.html page


# What to do:
# Conduct a linear regression, find closest precincts to that line and see what
# Those precincts are like...
# "Swing Precincts" eh that doesn't help much.


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from time import time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(
        ca_gov_reduc['democrat'], ca_gov_reduc['republican'], test_size=0.8, random_state=42)

plt.plot(X_train, y_train, 'b.')

def exp_mean_squared_error(estimator, X, y):
    y_predicted = np.exp(estimator.predict(X))
    y_true = np.exp(y)
    return -np.sqrt(mean_squared_error(y_true, y_predicted))

param_dist = {"max_depth": randint(1, 11),
              "learning_rate": uniform(),
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": randint(2, 50),
              "min_samples_leaf": randint(1, 11),
              "n_estimators": randint(50,150),
              "criterion": ["friedman_mse"]}

n_iter_search = 100

regressor = GradientBoostingRegressor(verbose = 1)

random_search = RandomizedSearchCV(regressor, 
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   scoring = exp_mean_squared_error,
                                   random_state = 5150)
start = time()

X_train_ln = np.array(X_train.map(lambda x: np.log(x) if x>0 else np.log(1)))
y_train_ln = np.array(y_train.map(lambda x: np.log(x) if x>0 else np.log(1)))

random_search.fit(X_train_ln.reshape(-1, 1), y_train_ln)

end = time()
print("total time: {} minutes".format((end-start)/60))



# Feature importance
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# Models
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, BayesianRidge

# Hyper parameter
from scipy.stats import randint as sp_randint
# from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from operator import itemgetter
# from sklearn import cross_validation

# Model Validation
#
#
#f_regression(X_train.reshape(-1,1), y_train.values.ravel())
#
#F, pval = f_regression(np.array(X_train).reshape(-1,1), y_train.values.ravel())
#Whatever

random_search.fit(X_train_ln.reshape(-1, 1), y_train_ln)

-random_search.best_score_<np.std(np.exp(y_train_ln))
-np.std(np.exp(y_train_ln))

plt.plot(np.exp(random_search.best_estimator_.predict(X_train_ln[:1000].reshape(-1,1))))

efficiency_ratio = expected/X_train_ln

# Exploration into ca_45 Mimi Walters vs

ca_45 = df[(df['office'] == "US House") & (df['district'] == "45")]

unravel = ca_45.set_index(['office', 'precinct', 'county', 'party'])['votes'].unstack()
unravel = unravel.reset_index()

ca_49 = df[(df['office'] == "US House") & (df['district'] == "49")]
unravel = ca_45.set_index(['office', 'precinct', 'county', 'party'])['votes'].unstack()

#A means vote by mail ballots

# Combine with income race blah blah datasets for classification algos? Or regression? eh?
