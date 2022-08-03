# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:48:19 2022

This code creates a chart plot with the accuracy metrics of each
combination of predictors

@author: rsrg_javier
"""

import matplotlib.pyplot as plt
import pandas as pd

cd C:\Users\rsrg_javier\Desktop\SEBAS\Manuscripts\2022\July\accuracy table

# read the table with the accuracy metrics
df = pd.read_excel('accuracy tables.xlsx', sheet_name='biomass RF')

df= df.set_index('Predictors')

# We can intruduce min, median and max manually
#plt.boxplot([[1, 2, 3], [3,4,5]])

df['r2 up'] = df['r2']+df['rsd']
df['r2 lo'] = df['r2']-df['rsd']

df['rrmse up'] = df['RRMSE']+df['RRMSEsd']
df['rrmse lo'] = df['RRMSE']-df['RRMSEsd']

df['srmse up'] = df['sRMSE']+df['sRMSEsd']
df['srmse lo'] = df['sRMSE']-df['sRMSEsd']

df['urmse up'] = df['uRMSE']+df['RRMSEsd']
df['urmse lo'] = df['uRMSE']-df['uRMSEsd']

print(plt.style.available)
plt.style.use('seaborn-colorblind')

font = 22

def boxplots(df):
    counter=0
    for row in range(0,len(df)):
        counter+=1
        row=df.iloc[(counter-1):counter]
    
        fig, ax1 = plt.subplots()
        props = dict(widths=0.7,patch_artist=True, medianprops=dict(color="black"))
        box1=ax1.boxplot(
                        [
                         [row['r2 lo'][0], row['r2'][0], row['r2 up'][0]],
                         [row['rrmse lo'][0], row['RRMSE'][0], row['rrmse up'][0]]
                                          ],
                        positions=[0,1], **props
                        )
        
        # Create a twin Axes sharing the xaxis.
        ax2 = ax1.twinx()
        box2=ax2.boxplot(
                        [
                        [row['srmse lo'][0], row['sRMSE'][0], row['srmse up'][0]],
                         [row['urmse lo'][0], row['uRMSE'][0], row['urmse up'][0]]
                                          ],
                         
                         
                         positions=[2,3], **props)
        ax2.set_ylim(30,80)
        #ax2.set_ylabel('$(g/m^{2})$', fontsize=font)


        # limit axes
        ax1.set_xlim(-0.5,3.5)
        ax1.set_ylim(0,1)
        ax1.set_xticks(range(0,4))       
        
        ax1.set_title(row.index[0], fontsize=font)
        ax1.set_xticklabels(['r2', 'RRMSE', 'sRMSE', 'uRMSE'], fontsize=font)
        #ax1.set_ylabel('%', fontsize=font)
        
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontsize(font)
            
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(font)
        
        # set colors with cycler
        for b in box1["boxes"]+box2["boxes"]:
            b.set_facecolor(next(ax1._get_lines.prop_cycler)["color"])
        
        plt.show()
        
        fname=f'{row.index[0]}.svg'
        fig.savefig(fname)


boxplots(df)


# TODO

fig, ax = plt.subplots(2,3)

