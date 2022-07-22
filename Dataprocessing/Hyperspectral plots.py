# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:49:10 2022

@author: rsrg_javier
"""

cd C:\Users\rsrg_javier\Desktop\SEBAS\Manuscripts\2022\July

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##############################################################################
# Read and process data 
##############################################################################
df = pd.read_csv('Hyperspectral_Sebas.csv')

# Create a column with the month

df['month'] = df.Date.str.extract(r'/([^\.]*)/', expand=False)

#df['month'] = list(map(lambda x : re.search(r'/(.+?)/', text).group(1), df['Date']))

df['month'] = df['month'].astype(int)

band = '832 nm'

# In case we want to use a ratio.
# I can use a ratio here, because I have all the values
# But I cannot display the sd histogram for a ratio with Sentinel-2 data
# Because I only have the median and sd, not each pixel values.
# Sentinel-2 data should be normalized perhaps.
# band2 = '680 nm'
# df['ratio'] = df[f'{band2}']/df[f'{band}']
# band = 'ratio'


# Pivot table per plot visitted
dfp = pd.pivot_table(df, 
                     values = band, 
                     index = ['PlotID','month', 'year'], 
                     aggfunc=['mean', 'std'])


# We have to drop a level in the columns index and rename it
cols =[f'{band} m',f'{band} sd']
dfp.columns = dfp.columns.droplevel()
dfp.columns = cols


# Lazy way to get a sorting field
dfp = dfp.sort_values(by =f'{band} sd').reset_index().reset_index()


# Create upper and lower boundaries with the std
dfp[f'{band} upper'] = dfp[f'{band} m']+dfp[f'{band} sd']
dfp[f'{band} lower'] = dfp[f'{band} m']-dfp[f'{band} sd']

##############################################################################
# Line plot of sorted nir values with standard deviation
##############################################################################
fig = sns.lineplot(data=dfp, x='index', y=f'{band} m')
plt.fill_between(x=dfp['index'], 
                 y1=dfp[f'{band} lower'], 
                 y2=dfp[f'{band} upper'], 
                 alpha=0.2, 
                 color='green')
plt.xlabel(f'Observations sorted by std of {band} values')
plt.ylabel(f'{band}')

#plt.savefig('nir median with sd.svg')# plot without the line

##############################################################################
# Plot histogram of standard deviation
##############################################################################
sns.histplot(dfp[f'{band} sd']*100, binwidth= 0.50, kde=True)
plt.xlabel('standard deviation of nir reflectance (%) from ASD fieldspec')
plt.axvline(dfp[f'{band} sd'].median()*100, color='red')



