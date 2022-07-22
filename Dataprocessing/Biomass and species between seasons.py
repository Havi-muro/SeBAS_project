# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:46:16 2022

@author: rsrg_javier

This script joins species and biomass data from SeBAS campaigns

It also divides the dataframe by campaign. 
This is done to compare variations in biomass and species between seasons
"""

cd C:\Users\rsrg_javier\Desktop\SEBAS\Fieldwork

import pandas as pd

df = pd.read_excel('SEBAS_FieldData_2020-2021_20042022_z.xlsx', sheet_name=0)

sp = pd.read_excel('SEBAS_Inventories_SpecRich_LS.xlsx', sheet_name=0)

complete = df.merge(sp, how='outer', on=['Qnum', 'month', 'year'])

complete = complete.drop(['Useful_PlotID_y', 'Exploratory_y', 'Date'], axis=1)


complete.to_csv('SEBAS_FieldData_2020-2021_16052022_spprich.csv')


comp = pd.pivot_table(complete, columns=['Useful_PlotID_x', 'month', 'year'], aggfunc='mean')


compT = comp.transpose()

compT = compT[['Biomass_gm2', 'SpecRich']]

compT = compT.reset_index()

compJ = compT[compT.month == 6]
J20 = compJ[compJ.year==2020]
J21 = compJ[compJ.year==2021]

compS = compT[compT.month == 9]
S20 = compS[compS.year==2020]
S21 = compS[compS.year==2021]


y2020 = J20.merge(S20, how='inner', on =['Useful_PlotID_x', 'year'])
y2021 = J21.merge(S21, how='inner', on =['Useful_PlotID_x', 'year'])


y2020['sub_biom'] = y2020['Biomass_gm2_x']-y2020['Biomass_gm2_y']
y2021['sub_biom'] = y2021['Biomass_gm2_x']-y2021['Biomass_gm2_y']

y2020['sub_spp'] = y2020['SpecRich_x']-y2020['SpecRich_y']
y2021['sub_spp'] = y2021['SpecRich_x']-y2021['SpecRich_y']

tot = pd.concat([y2020, y2021], axis=0)

tot = tot.rename(columns={'Biomass_gm2_x':'biomass_gm2_june', 'Biomass_gm2_y': 'biomass_gm2_sep',
                          'SpecRich_x':'SpecRich_june', 'SpecRich_y': 'SpecRich_sep'})

tot.to_csv('SeBAS_biomass_substraction2.csv')
