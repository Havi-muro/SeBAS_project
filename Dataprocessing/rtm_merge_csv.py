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


cd D:\SeBAS_RS\RS\RTM_GEE\LAI

ls

myls = glob('*.csv')

ls=[]
for i in range(len(myls)):
    df = pd.read_csv(myls[i])
    print(df.head())
    ls.append(df)
    
tot = pd.concat(ls)
tot=tot.drop('Unnamed: 0', axis=1)

sebas = pd.read_csv('C:/Users/rsrg_javier/Desktop/SEBAS/Fieldwork/Data4bexis/Copy of SEBAS_FieldData_2020-2021_09082022_cf.csv')

tut = sebas.merge(tot, left_on=['Qnum', 'year', 'month'], right_on=['Plot_ID', 'year', 'month'])

tut['Brown_veg_perc_cover'] = tut['Senescent_percent_cover']+tut['Moribund_percent_cover']

# make a list with the sch campaign where LAI readings are too high
tut2=tut[(tut['Exploratory']=='SCH') & (tut['month'] == 9) & (tut['year'] == 2021)]

# remove those rows and seelct relevant columns
tut3 = pd.merge(tut, tut2, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)

mycols = ['Useful_PlotID', 'Exploratory', 'Qnum', 'month', 'year', 
          'LAI_field', 'Green_veg_percent_cover',
          'Brown_veg_perc_cover', 'LAI_rtm']
tut4 = tut3[mycols]
tut5 = tut4.dropna()

# Simple scatter plot
sns.scatterplot(data=tut5, 
                x='LAI_field',
                y='LAI_rtm')

# Density plot
y = tut5['LAI_rtm']
x = tut5['LAI_field']

# Calculate the point density
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=50)


plt.ylabel(f'LAI RTM')
plt.xlabel(f'In situ LAI')

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

