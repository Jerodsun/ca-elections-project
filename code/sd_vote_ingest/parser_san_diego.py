# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:36:28 2019

@author: Jerod
"""

import pandas as pd
import matplotlib.pyplot as plt

df_2014 = pd.read_csv('../../raw_resources/san_diego/201411cv.txt', sep='\t')
df_2014 = df_2014.rename(columns = {'County_id':'county_id','Consolidation_Name': 'consolidation_name', 
                          'contest-stats':'stats_contest', 'Contest_id-stat_id': 'contest_id',
                          'candidate-stat':'stats_candidate', 'candidate_id-stat_id': 'candidate_id_stat_id',
                          'counttype':'count_type', 'value':'stats_votes'})
# standardize formats w/2016 format

df_2014 = df_2014[~df_2014['stats_candidate'].isin(['Times Counted', 'Number of Precincts for Race'])]
df_2014 = df_2014[df_2014['count_type'] == "Total"] # keep mail and poll separate for further analysis



df_2016 = pd.read_csv('../../raw_resources/san_diego/20161108cv.txt', sep='\t')
df_2016 = df_2016[df_2016['count_type'] == "Total"] # keep mail and poll separate for further analysis
df_2016 = df_2016[~df_2016['stats_candidate'].isin(['Times Counted', 'Number of Precincts Reporting'])]


df_2018 = pd.read_csv('../../raw_resources/san_diego/20181106cv.txt', sep=',', header=None, encoding='latin-1')
df_2018 = df_2018[df_2018[20] == "Total"]
df_2018 = df_2018.rename(columns={0:'county_id', 2:'consolidation_name', 3:'consolidation_id',
                        5:'stats_contest', 6:'contest_id', 14:'stats_candidate', 15:'contest_id_cand_id', 
                        17:'party', 20:'count_type', 22:'stats_votes'})

df_2018 = df_2018[['county_id', 'consolidation_name', 'consolidation_id',
       'stats_contest', 'contest_id', 'stats_candidate',
       'contest_id_cand_id', 'party', 'count_type', 'stats_votes']]

df_2018 = df_2018[~df_2018['stats_contest'].isin(['Race Statistics'])]
df_2018 = df_2018[~df_2018['stats_candidate'].isin(['Times Counted', 'Number of Precincts for Race', 'Number of Precincts Reporting'])]



df_2014_sen_77 = df_2014[df_2014['stats_contest'] == 'STATE ASSEMBLY-77TH DIST.']
mask_2014_sen_77 = set(df_2014_sen_77['consolidation_name'])
df_2014_selection = df_2014[df_2014['consolidation_name'].isin(mask_2014_sen_77)]


df_2016_sen_77 = df_2016[df_2016['stats_contest'] == 'STATE ASSEMBLY - 77th District']
mask_2016_sen_77 = set(df_2014_sen_77['consolidation_name'])
df_2016_selection = df_2016[df_2016['consolidation_name'].isin(mask_2016_sen_77)]


df_2018_sen_77 = df_2018[df_2018['stats_contest'] == 'STATE ASSEMBLY 77TH DIST']
mask_2018_sen_77 = set(df_2018_sen_77['consolidation_name'])
df_2018_selection = df_2018[df_2018['consolidation_name'].isin(mask_2018_sen_77)]
#df_2014_selection_2 = df_2014_selection[df_2014_selection['stats_contest'].isin(['STATE ASSEMBLY-77TH DIST.', 'STATE BOARD OF EQUALIZATION 4TH DISTRICT'])]


def reduce(df_slice):
    """ Calls unstack() to combine rows into columns """
    df_slice
    result = df_slice.set_index(['county_id', 'consolidation_name', 'consolidation_id',
       'stats_contest', 'contest_id', 
       'stats_candidate'])['stats_votes'].unstack().reset_index()
    return result.reset_index()


init_2014 = reduce(df_2014_selection[df_2014_selection['stats_contest'] == 'STATE ASSEMBLY-77TH DIST.'])
init_2016 = reduce(df_2016_selection[df_2016_selection['stats_contest'] == 'STATE ASSEMBLY - 77th District'])
init_2018 = reduce(df_2018_selection[df_2018_selection['stats_contest'] == 'STATE ASSEMBLY 77TH DIST'])


init_2014 = reduce(df_2014_selection[df_2014_selection['stats_contest'] == 'STATE ASSEMBLY-77TH DIST.'])
for i in ['STATE BOARD OF EQUALIZATION 4TH DISTRICT']:
    df_reduc = reduce(df_2014_selection[df_2014_selection['stats_contest'] == i])
    df_reduc = df_reduc.drop(columns=['contest_id', 'stats_contest'])
    init_2014 = init_2014.merge(df_reduc, on=["county_id", "index", "consolidation_name", "consolidation_id",
                                              "Registered Voters"])
    
plt.scatter(init_2014["BRIAN  MAIENSCHEIN"]/init_2014['RUBEN HERNANDEZ'] , init_2014['DIANE L. HARKEY'] / init_2014['NADER SHAHATIT'])


#ok now to the big question: 2018 race characteristics Insurance Commissioner

init_2018_1 = reduce(df_2018[df_2018['stats_contest'] == 'GOVERNOR'])

for i in ['INSURANCE COMMISSIONER', 'STATE BOARD OF EQUALIZATION 4TH DISTRICT']:
    df_reduc = reduce(df_2018[df_2018['stats_contest'] == i])
    df_reduc = df_reduc.drop(columns=['contest_id', 'stats_contest'])
    init_2018_1 = init_2018_1.merge(df_reduc, on=["county_id", "index", "consolidation_name", "consolidation_id",
                                              "Registered Voters"])
    
init_2018_1_l = init_2018_1[init_2018_1['Registered Voters'] > 50]
init_2018_1_l = init_2018_1_l.drop([2136])

init_2018_1_l['gub'] = init_2018_1_l['GAVIN NEWSOM'] + init_2018_1_l['JOHN H. COX']
init_2018_1_l['eq'] = init_2018_1_l['JOEL  ANDERSON'] + init_2018_1_l['MIKE SCHAEFER']
init_2018_1_l['insur'] = init_2018_1_l['RICARDO LARA'] + init_2018_1_l['STEVE POIZNER']
init_2018_1_l['gub_turnout'] = init_2018_1_l['gub'] / init_2018_1_l['Registered Voters']


plt.scatter(init_2018_1_l["GAVIN NEWSOM"], init_2018_1_l['JOHN H. COX'])
plt.scatter(init_2018_1_l['RICARDO LARA'], init_2018_1_l['STEVE POIZNER'])

plt.scatter(init_2018_1_l["GAVIN NEWSOM"]/init_2018_1_l['JOHN H. COX'], init_2018_1_l['RICARDO LARA']/init_2018_1_l['STEVE POIZNER'])

all_2014 = set(df_2014['consolidation_name'])
all_2016 = set(df_2016['consolidation_name'])
all_2018 = set(df_2018['consolidation_name'])
print(len(all_2014), len(all_2016), len(all_2018))
len(all_2014 & all_2016 & all_2018)
# shifts from 2014 to 2018... that's the interesting part. total delta... requires combining tracts...

# All for San Diego only


sort = init_2018_1_l.sort_values(by=['gub','eq', 'insur']).reset_index(drop=True)
# comment out for another run
sort = sort[sort['gub'] > 1000]

plt.scatter(sort.index, sort['gub'])
plt.scatter(sort.index, sort['eq'])
plt.scatter(sort.index, sort['insur'])
plt.legend(['Governor','State Board of Equalization', 'Insurance Commissioner'])
plt.ylabel("Total Votes")
plt.xticks([])

# Visualize voter drop off

sort = init_2018_1_l.sort_values(by=['gub_turnout','eq', 'insur']).reset_index(drop=True)
plt.scatter(sort.index, sort['gub_turnout']) # CDF

sort['gub_turnout'].hist(bins=50)

import numpy as np
from scipy.stats import norm

# Generate some data for this demonstration.
# Fit a normal distribution to the data:
mu, std = norm.fit(sort['gub_turnout'])

# Plot the histogram.
plt.hist(sort['gub_turnout'], bins=25, density=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.show()