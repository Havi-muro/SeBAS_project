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

 
indir = os.path.join('C:\\','Users','rsrg_javier','Desktop','SEBAS','Temporal crossvalidation' )

# Merge the old dataframe with the new biomass
# sebasold = pd.read_csv('C:/Users/rsrg_javier/Desktop/SEBAS/Fieldwork/Data4bexis/SEBAS_FieldData_2020-2021_22072022.csv')
# sebas = pd.read_csv(indir + '\\sebas_biomass.csv')
# sebas['month'].replace(regex=True, inplace=True, to_replace=5, value=6)
# sebasnew = sebasold.merge(sebas, on =['Qnum', 'month', 'year'], how='outer')
# sebasnew.to_csv(indir+'\sebas_new_biomass.csv')

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

# Create a column with Useful_PlotID
survey['Useful_PlotID'] = survey['EP_PlotID']
survey['EG'] = survey['EP_PlotID']

# get the plot number and fill it with leading 0s
survey['EG'] = survey['EG'].str.slice(start=3)
survey['EG'] = survey['EG'].str.zfill(2)

# get the other field
survey['Useful_PlotID'] = survey['Useful_PlotID'].str.slice(stop=3)

# Concatenate them
survey['Useful_PlotID'] = survey['Useful_PlotID'] + survey['EG']
survey = survey.drop(['EG','EP_PlotID'] , axis=1)
survey = survey.rename(columns={'Year':'year'})


sebasdc = sebas_s2.merge(survey, on=['Useful_PlotID', 'year'], how='left')

# Get the dates right

# infer_datetime_format=True
sebasdc['date_releves']= pd.to_datetime(sebasdc['date_releves'], format = '%d.%m.%Y').dt.strftime('%Y-%m-%d')
sebasdc['S2_timestep']= pd.to_datetime(sebasdc['S2_timestep'], format = '%Y%m%d').dt.strftime('%Y-%m-%d')
sebasdc['DateCut1'] =    pd.to_datetime(sebasdc['DateCut1'], format = '%Y-%m-%d', errors='coerce')
sebasdc['DateCut2'] =    pd.to_datetime(sebasdc['DateCut2'], format = '%Y-%m-%d', errors='coerce')
sebasdc['DateCut3'] =    pd.to_datetime(sebasdc['DateCut3'], format = '%Y-%m-%d', errors='coerce')

sebass = sebasdc[['Qnum', 'year', 'month', 'date_releves', ]]


emdf = pd.DataFrame()

def setdates (df,r,s2,dc1,dc2,dc3):
    for row in range(0,len(df)):
        if r<dc1<s2:
            df.drop(row, axis=0)
        elif s2<dc1<r:
            df.drop(row, axis=0)
    return df

def setdates (df,r,s2,dc1,dc2,dc3):
    for row in range(0,len(df)):
        if r<dc1<s2:
            emdf.append(row)
        elif s2<dc1<r:
            df.drop(row, axis=0)
    return df

def setdates (df,r,s2,dc1,dc2,dc3):
    for row in range(0,len(df)):
        if r<dc1<s2:
            print(row)



result = setdates(sebasdc,
                  sebasdc['date_releves'], 
                  sebasdc['S2_timestep'], 
                  sebasdc['DateCut1'], 
                  sebasdc['DateCut2'],
                  sebasdc['DateCut3'])
        
resta =dt(sebasdc['date_releves'][1]) - sebasdc['S2_timestep'][1]

sebasdc['DateCut1'][1]

drop = sebasdc.drop(3, axis=0)



