# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:52:37 2022

read survey data
read sebas data
combine both, and set the dates format
eliminate all records in which mowing event took place between vegetation
releves and S2 image aquisition

@author: rsrg_javier
"""
import os
import pandas as pd
import datetime as dt
from datetime import datetime

 
indir = os.path.join('C:\\','Users','Janny','Desktop','SEBAS','Tempcv' )


# Read the sebas data and survey data
sebas_s2 = pd.read_csv(indir+ '\sebas_new_biomass.csv')
survey0 = pd.read_csv(indir+ '\\26487_48_data.csv', sep=';')


list(survey0.columns)

# Select the columns and years we need
mycols = [
 'Year',
 'EP_PlotID','DateCut1',
 'DateCut2',
 'DateCut3']

survey = survey0[mycols]

survey = survey[survey['Year'] >= 2020]

# Create two columns to form Useful_PlotID
survey['Useful_PlotID'] = survey['EP_PlotID']
survey['plotn'] = survey['EP_PlotID']

# get the plot number and fill it with leading 0s
survey['plotn'] = survey['plotn'].str.slice(start=3)
survey['plotn'] = survey['plotn'].str.zfill(2)

# get the EG field
survey['Useful_PlotID'] = survey['Useful_PlotID'].str.slice(stop=3)

# Concatenate them
survey['Useful_PlotID'] = survey['Useful_PlotID'] + survey['plotn']
survey = survey.drop(['plotn','EP_PlotID'] , axis=1)
survey = survey.rename(columns={'Year':'year'})

sebasdc = sebas_s2.merge(survey, on=['Useful_PlotID', 'year'], how='left')

# Get the dates right
sebasdc['date_releves']= pd.to_datetime(sebasdc['date_releves'], format = '%d.%m.%Y')#.dt.strftime('%Y-%m-%d')
sebasdc['date_uav']= pd.to_datetime(sebasdc['date_uav'], format = '%d.%m.%Y')#.dt.strftime('%Y-%m-%d')

sebasdc['S2_timestep']= pd.to_datetime(sebasdc['S2_timestep'], format = '%Y%m%d')#.dt.strftime('%Y-%m-%d')
sebasdc['DateCut1'] =    pd.to_datetime(sebasdc['DateCut1'], format = '%Y-%m-%d', errors='coerce')#.dt.strftime('%Y-%m-%d')
sebasdc['DateCut2'] =    pd.to_datetime(sebasdc['DateCut2'], format = '%Y-%m-%d', errors='coerce')#.dt.strftime('%Y-%m-%d')
sebasdc['DateCut3'] =    pd.to_datetime(sebasdc['DateCut3'], format = '%Y-%m-%d', errors='coerce')#.dt.strftime('%Y-%m-%d')
sebasdc = sebasdc.drop('Unnamed: 0', axis=1)

sebasdc.to_csv(indir+'\\sebas_df_harvestdates.csv')


# Collect all the plots that were harvested between
# field and image acquisition
def harvest (df):
    emdf = pd.DataFrame()
    for i in range(0,len(df)):
        row = df.loc[i]
        r = df.iloc[i,6]
        s2 = df.iloc[i,10]
        dc1 = df.iloc[i,39]
        dc2 = df.iloc[i,40]
        dc3 = df.iloc[i,41]     
        if r<dc1<s2:
            emdf = pd.concat([emdf, pd.Series(row)], axis=1)
        elif s2<dc1<r:
            emdf = pd.concat([emdf, pd.Series(row)], axis=1)
        if r<dc2<s2:
            emdf = pd.concat([emdf, pd.Series(row)], axis=1)
        elif s2<dc2<r:
            emdf = pd.concat([emdf, pd.Series(row)], axis=1)
        if r<dc3<s2:
            emdf = pd.concat([emdf, pd.Series(row)], axis=1)
        elif s2<dc3<r:
            emdf = pd.concat([emdf, pd.Series(row)], axis=1)
    return emdf.T
           
harvest = harvest(sebasdc)

# Eliminate the records of harvested from the original dataset
# We create a unique identifier
sebasdc['code'] = sebasdc['Qnum']+sebasdc['month'].astype(str)+sebasdc['year'].astype(str)
harvest['code'] = harvest['Qnum']+harvest['month'].astype(str)+harvest['year'].astype(str)

gooddf = sebasdc[~sebasdc.code.isin(harvest.code)]

gooddf.to_csv(indir+'\\df_temp_cv_harv.csv')


