# scatter plot of RCFM and FM colored by site
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as font_manager

# read the xlsx file
df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_forPython.xlsx', sheet_name='MatchedData')

# add OM_imposing with 2.5 threshold
df['OM_imposing'] = df.apply(lambda row: row['OM'] if row['OM/OC'] < 2.5 else row['OC']*2.5, axis=1)

# calculate RCFM and RCFM_imposing
df['RCFM'] = df['AS'] + df['AN'] + df['SS'] + df['Soil'] + df['EBC'] + df['TEO']*0.001 + df['OM']
df['RCFM_imposing'] = df['AS'] + df['AN'] + df['SS'] + df['Soil'] + df['EBC'] + df['TEO']*0.001 + df['OM_imposing']

# Write the updated DataFrame to a new Excel file
df.to_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_output.xlsx', index=False, sheet_name='RCFM_imposing')

# RCFM_imposing values are retained if FM exceeds the RCFM by less than 50%, otherwise values are flagged as underestimated and not reported.
# create a new column 'Flag' and flag the underestimated values
df['Flag'] = df['RCFM_imposing']
df.loc[df['FM'] > df['RCFM_imposing']*1.5, 'Flag'] = 'Underestimated'
# replace the underestimated values in the RCFM_imposing column with blank
df.loc[df['FM'] > df['RCFM_imposing']*1.5, 'RCFM_imposing'] = ''

# print the updated dataframe
print(df.head())

# Write the updated DataFrame to a new sheet in the Excel file
with pd.ExcelWriter('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_output.xlsx', mode='a') as writer:
    df.to_excel(writer, sheet_name = 'RCFM_Underestimated')