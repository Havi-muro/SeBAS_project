# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:29:30 2022

This codes gathers together the csv files with the rtm values for each campaign,
merges them with the sebas data, and plots relationships

@author: rsrg_javier
"""

from glob import glob
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import os

# Read the sebas field data
sebas = pd.read_csv('C:/Users/rsrg_javier/Desktop/SEBAS/Fieldwork/Data4bexis/Copy of SEBAS_FieldData_2020-2021_09082022_cf.csv')
sebas['Brown_veg_perc_cover'] = sebas['Senescent_percent_cover']+sebas['Moribund_percent_cover']

# read cwm data
allCWM = pd.read_csv('C:/Users/rsrg_javier/Desktop/SEBAS/Fieldwork/CWMs/sebas_cwm_traits.csv')

# merge cwm and sebas data
sebas = sebas.merge(allCWM, on=['Qnum', 'month', 'year'])

# merge all rtm traits with sebas data
# list traits
traitls = ['LAI', 'Cab', 'Cm', 'Cw']

for trait in traitls:
        
    os.chdir(os.path.join('D://','SeBAS_RS','RS','RTM_GEE',trait))
    
    #list files in each trait folder and concatenate them to a big df
    myls = glob('*.csv')
    
    ls2concat=[]
    for i in range(len(myls)):
        df = pd.read_csv(myls[i])
        #print(df.head())
        ls2concat.append(df)
        
    tot = pd.concat(ls2concat)
    tot=tot.drop('Unnamed: 0', axis=1)
    
    # we code the summer campaign with 6, eventhough sometimes it was end of May
    tot['month'].replace(regex=True,
                            inplace=True,
                            to_replace= 5,
                            value=6)
    # Merge trait and sebas data
    sebas = sebas.merge(tot, 
                      left_on=['Qnum', 'year', 'month'], 
                      right_on=['Plot_ID', 'year', 'month'],
                      how= 'left')
##############################################################################

# there seems to be 100 quadrats visited but without rtm LAI

# select relevant columns
mycols = ['Exploratory', 'Qnum', 'month', 'year', 
          'LAI_field', 'Green_veg_percent_cover','Brown_veg_perc_cover', 
          'Hveg_cwm', 'Hrep_cwm', 'SPAD_cwm', 'LDMC_cwm',
          'LA_cwm', 'SLA_cwm', 'rel_LWC_cwm', 'LWC_cwm', 'LMA_cwm',
          'LAI_rtm','laiCab_rtm', 'laiCm_rtm', 'laiCw_rtm']

sebas_cols = sebas[mycols]

sebas_cols.columns=['Exploratory', 'Qnum', 'month', 'year', 
          'LAI_cwm', 'Green_veg_percent_cover','Brown_veg_perc_cover', 
          'Hveg_cwm', 'Hrep_cwm', 'Cab_cwm', 'Cm_cwm',
          'LA_cwm', 'SLA_cwm', 'rel_Cw_cwm', 'Cw_cwm', 'LMA_cwm',
          'laiLAI_rtm','laiCab_rtm', 'laiCm_rtm', 'laiCw_rtm']

# make a list with the sch campaign where LAI readings are too high
schlai=sebas_cols[(sebas_cols['Exploratory']=='SCH') 
                  & (sebas_cols['month'] == 9) 
                  & (sebas_cols['year'] == 2021)]

# remove sch rows 
sebas_nosch = pd.merge(sebas_cols, schlai, 
                indicator=True, 
                how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

##############################################################################
# select trait to plot
for trait in traitls:
    toplot = sebas_nosch[[f'{trait}_cwm', f'lai{trait}_rtm']]
    toplot = toplot.dropna()
        
    # Density plot
    x=toplot.iloc[:,0]
    y=toplot.iloc[:,1]
    
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)
    
    plt.ylabel(f'{trait} RTM')
    plt.xlabel(f'In situ {trait}')
    
    #add a r=1 line
    line = np.array([0,max(y)])
    plt.plot(line,line,lw=1, c="black")
    plt.show()

# Density plot
toplot = sebas_nosch[['Cab_cwm', 'laiCab_rtm']]
toplot = toplot.dropna()

x=toplot.iloc[:,0]
y=toplot.iloc[:,1]*10

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=50)

plt.ylabel(f'{trait} RTM')
plt.xlabel(f'In situ {trait}')

#add a r=1 line
line = np.array([0,max(y)])
plt.plot(line,line,lw=1, c="black")
plt.show()




import plotly.express as px
from plotly.offline import plot

fig =px.scatter(data_frame= tut5, x=x, y=y, color='Senescent_veg_perc_cover', symbol='year', size='month')
fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                          ticks="outside",
                                         ))
plot(fig)


# check what happened in Hainich september 2021. Many underestimations of LAI,
# but image acquisitions is 08-sep, and field campaign was 13-17 sep

