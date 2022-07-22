# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:19:59 2022

This code aggregares the variable imortances from Shap by band or by date
and plots the results

@author: rsrg_javier
"""
cd C:\Users\rsrg_javier\Documents\GitHub\SeBAS_project\Dataprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# set style of plots
print(plt.style.available)
plt.style.use('seaborn')

# read data
df = pd.read_csv('SppRich_shap.csv')

# craete a column with the time step and the band to aggregate later
step = df['col_name'].str.split('_', expand=True)[[1]]
step = step.rename(columns={1:'step'})

band = df['col_name'].str.split('_', expand=True)[[0]]
band = band.rename(columns={0:'band'})
band['bandt']=band['band']

df = pd.concat([df, step, band], axis=1)

# sort and subset values for testing
df1 = df.sort_values('shap', ascending=False)
df1 = df1.head(10)

plt.bar(range(len(df1['shap'])), 
        df1['shap'], 
        yerr=df1['std'], 
        alpha=0.2, 
        align='center'
        )
plt.xticks(range(len(df1['shap'])), df1['date'],rotation=20)
plt.show()


#############################################################################
# aggregate data band    
#############################################################################

byband = pd.pivot_table(df, values=['shap', 'std', 'band'], 
                        index='bandt', 
                        aggfunc ={'shap':'mean', 'std':'mean', 'band':'first'}).reset_index()

# create an indexed df with the bands in the right order and merge it
keys = pd.DataFrame({'band':['blue','green','red','re1','re2','re3','nir','nirb','swir1','swir2']})
bybandm = keys.merge(byband, on='band').set_index('band')

# plot the barchart
# x is a sequence of scalars representing the x coordinates of the bars
fig = plt.figure(figsize=(10,5))
plt.bar(range(len(bybandm['bandt'])), bybandm['shap'], yerr=bybandm['std'], color='cadetblue')
plt.xticks(range(len(bybandm['shap'])), bybandm['bandt'],rotation=30, size=20)
plt.yticks(fontsize=20)
plt.title('by band',size=20)
plt.ylabel('Shap values', size=20)
plt.show()
fig.savefig('shap_spprich_bands.svg', format='svg')


#############################################################################
# Analyze by date with fewer bands  
#############################################################################

dfs = df[df.band != 're3']
dfs = dfs[dfs.band != 'blue']
dfs = dfs[dfs.band != 'red']
dfs = dfs[dfs.band != 'green']

bydate = pd.pivot_table(df, values=['shap', 'std', 'date'], 
                        index='step', 
                        aggfunc ={'shap':'mean', 'std':'mean', 'date':'first'}).reset_index()

# set step field as integer and make it the index
bydate['step']=bydate['step'].astype(int)
bydate=bydate.sort_values('step').set_index('step')

# plot the barchart
# x is a sequence of scalars representing the x coordinates of the bars
fig = plt.figure(figsize=(10,5))

plt.bar(range(len(bydate['date'])), bydate['shap'], yerr=bydate['std'], color='mediumseagreen')
plt.xticks(range(len(bydate['shap'])), bydate['date'],rotation=75, size=20)
plt.yticks(size=20)
plt.title('by date', size=20)
plt.ylabel('Shap values', size=20)

plt.show()
# Export to vector
fig.savefig('shap_spprich_date.svg', format='svg')


#############################################################################
# Biomass  
#############################################################################

df = pd.read_csv('Biomass_shap.csv')

fig = plt.figure(figsize=(15,5))
plt.bar(range(len(df['shap'])), 
        df['shap'], 
        yerr=df['std'], 
        align='center',
        color='teal'
        )
plt.xticks(range(len(df['shap'])), df['Predictor'],rotation=75, size=20)
plt.yticks(fontsize=20)
plt.title('All predictors', size=20)
plt.ylabel('Shap values', size=20)

plt.show()
fig.savefig('shap_biomass_predictors.svg', format='svg')

# chart by bands

df = df[(df.Predictor == 'blue') | 
        (df.Predictor == 'green') |
        (df.Predictor == 'red') |
        (df.Predictor == 'nir') |
        (df.Predictor == 'nirb') |
        (df.Predictor == 're1') |
        (df.Predictor == 're2') |
        (df.Predictor == 're3') |
        (df.Predictor == 'swir1') |
        (df.Predictor == 'swir2')
        ]
keys = pd.DataFrame({'Predictor':['blue','green','red','re1','re2','re3','nir','nirb','swir1','swir2']})
dfm = keys.merge(df, on='Predictor').set_index('Predictor')

fig = plt.figure(figsize=(15,5))
plt.bar(range(len(df['shap'])), 
        df['shap'], 
        yerr=df['std'], 
        align='center',
        color='sienna'
        )
plt.xticks(range(len(df['shap'])), df['Predictor'],rotation=50, size=20)
plt.yticks(fontsize=20)
plt.title('S2 bands', size=20)
plt.ylabel('Shap values', size=20)

plt.show()
fig.savefig('shap_biomass_bands.svg', format='svg')




















