import pandas as pd
import os

# Read the data
other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
IMPROVE_df = pd.read_excel(other_obs_dir + 'IMPROVE_EC_2019_raw.xlsx', sheet_name='Data')

# Convert 'Date' column to datetime format
IMPROVE_df['Date'] = pd.to_datetime(IMPROVE_df['Date'], format='%m/%d/%Y')
IMPROVE_df['Year'] = IMPROVE_df['Date'].dt.year
IMPROVE_df['Month'] = IMPROVE_df['Date'].dt.month
IMPROVE_df['Day'] = IMPROVE_df['Date'].dt.day
# Filter data for the year 2019
IMPROVE_df = IMPROVE_df[IMPROVE_df['Year'] == 2019]
# Filter valid data, Flag = V0
IMPROVE_df = IMPROVE_df[IMPROVE_df['ECf_Status'] == 'V0']
# Filter POC = 1
IMPROVE_df = IMPROVE_df[IMPROVE_df['POC'] == 1]

# Group by 'SiteName', 'Latitude', 'Longitude', and calculate monthly average 'ECf_Val'
EC_mon_df = IMPROVE_df.groupby(['SiteName', 'Latitude', 'Longitude', 'Month'])['ECf_Val'].agg(['count', 'mean']).reset_index()
EC_mon_df.columns = ['SiteName', 'Latitude', 'Longitude', 'Month', 'num_obs', 'EC_mon']

# QC: print rows with num_obs > 11
rows_with_more_than_11_obs = EC_mon_df[EC_mon_df['num_obs'] > 11]
if not rows_with_more_than_11_obs.empty:
    print(rows_with_more_than_11_obs)
else:
    print("No rows with > 11 observations.")

# Save EC_mon_df as sheet 'mon'
with pd.ExcelWriter(os.path.join(other_obs_dir + 'IMPROVE_EC_Summary.xlsx'), engine='openpyxl', mode='w') as writer:
    EC_mon_df.to_excel(writer, sheet_name='mon', index=False)

# Calculate the annual average 'ECf_Val' for each 'SiteName', 'Latitude', 'Longitude'
EC_annual_df = EC_mon_df.groupby(['SiteName', 'Latitude', 'Longitude'])['EC_mon'].mean().reset_index()
EC_annual_df.columns = ['SiteName', 'Latitude', 'Longitude', 'EC_annual']

# Save the annual DataFrame as 'annual'
with pd.ExcelWriter(os.path.join(other_obs_dir + 'IMPROVE_EC_Summary.xlsx'), engine='openpyxl', mode='a') as writer:
    EC_annual_df.to_excel(writer, sheet_name='annual', index=False)
