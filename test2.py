#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
from gamap_colormap import WhGrYlRd
import calendar
import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.io import loadmat
import matplotlib.lines as mlines

cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}-{}/monthly/'.format(cres.lower(), deposition) # CEDS, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower()) # HTAP, LUO
# sim_dir = '/Volumes/rvmartin/Active/dandan.z/AnalData/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year) # C720, HTAP, LUO
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
FTIR_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/raw_data_and_reports/'
################################################################################################
# Extract BC_HIPS from masterfile and lon/lat from site.details
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(obs_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350', 'MXMC-0006', 'NGIL-0309'
    ]
    HIPS_FTIR_dfs = []
    for filename in os.listdir(obs_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')
            HIPS_FTIR_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                            'BC_HIPS_ug', 'EC_FTIR_ug', 'Flags']
            if all(col in master_data.columns for col in HIPS_FTIR_columns):
                # Select the specified columns
                master_data.columns = master_data.columns.str.strip()
                HIPS_FTIR_df = master_data[HIPS_FTIR_columns].copy()
                # Exclude specific FilterID values
                HIPS_FTIR_df = HIPS_FTIR_df[~HIPS_FTIR_df['FilterID'].isin(excluded_filters)]
                # Select PM2.5
                HIPS_FTIR_df['Mass_type'] = pd.to_numeric(HIPS_FTIR_df['Mass_type'], errors='coerce')
                HIPS_FTIR_df = HIPS_FTIR_df.loc[HIPS_FTIR_df['Mass_type'] == 1]
                # Convert the relevant columns to numeric
                HIPS_FTIR_df[['BC_HIPS_ug', 'EC_FTIR_ug', 'mass_ug', 'Volume_m3', 'start_year', 'start_month', 'start_day']] = HIPS_FTIR_df[
                    ['BC_HIPS_ug', 'EC_FTIR_ug', 'mass_ug', 'Volume_m3', 'start_year', 'start_month', 'start_day']].apply(pd.to_numeric, errors='coerce')
                # Select year 2019 - 2023
                # HIPS_df = HIPS_df[HIPS_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]
                # Drop rows with NaN values
                HIPS_FTIR_df = HIPS_FTIR_df.dropna(subset=['start_year', 'Volume_m3'])
                HIPS_FTIR_df = HIPS_FTIR_df[HIPS_FTIR_df['Volume_m3'] > 0]
                # HIPS_df = HIPS_df[HIPS_df['BC_HIPS_ug'] > 0]
                # Calculate BC concentrations
                HIPS_FTIR_df.rename(columns={'EC_FTIR_ug': 'EC_FTIR_ug_master'}, inplace=True)
                HIPS_FTIR_df['BC_HIPS_ug/m3'] = HIPS_FTIR_df['BC_HIPS_ug'] / HIPS_FTIR_df['Volume_m3']
                HIPS_FTIR_df['EC_FTIR_ug/m3_master'] = HIPS_FTIR_df['EC_FTIR_ug_master'] / HIPS_FTIR_df['Volume_m3']
                HIPS_FTIR_df['PM25_ug/m3'] = HIPS_FTIR_df['mass_ug'] / HIPS_FTIR_df['Volume_m3']
                # Extract the site name and add as a column
                site_name = filename.split('_')[0]
                HIPS_FTIR_df["Site"] = [site_name] * len(HIPS_FTIR_df)
                # Append the current HIPS_df to the list
                HIPS_FTIR_dfs.append(HIPS_FTIR_df)
            else:
                print(f"Skipping {filename} because not all required columns are present.")
    return pd.concat(HIPS_FTIR_dfs, ignore_index=True)

# Function to read and preprocess UV-Vis data
def read_UV_Vis_files(out_dir):
    UV_df = pd.read_excel(os.path.join(out_dir, 'BC_UV-Vis_SPARTAN/BC_UV-Vis_SPARTAN_Joshin_20230510.xlsx'),
                          usecols=['Filter ID', 'f_BC', 'Location ID'])
    # Drop the last two digits in the 'Filter ID' column: 'AEAZ-0113-1' to 'AEAZ-0113'
    UV_df['FilterID'] = UV_df['Filter ID'].str[:-2]
    UV_df.rename(columns={'Location ID': 'Site'}, inplace=True)
    # Convert columns to numeric
    UV_df['f_BC'] = pd.to_numeric(UV_df['f_BC'], errors='coerce')
    # Exclude values where f_BC > 0.8
    UV_df = UV_df[UV_df['f_BC'] <= 0.8]
    return UV_df

# Function to read and preprocess raw FTIR data
def read_FTIR_raw_data(FTIR_dir):
    # Read the two Excel files
    raw_batch4_df = pd.read_excel(FTIR_dir + '04_SPARTAN_FTIR data_batch 4_ Nov2022 to March2024.xlsx', header=1)
    raw_batch2and3_df = pd.read_excel(FTIR_dir + '04_SPARTAN_FTIR data batch 2 and 3 resubmitted with MDLs.xlsx',
                                      header=1)
    raw_batch1_df = pd.read_excel(FTIR_dir + '01_SPARTAN_FTIR_and_FG_predictions_08032022.xlsx',
                                      header=1)
    # Standardize the 'date' column
    raw_batch4_df['date'] = pd.to_datetime(raw_batch4_df['date'], errors='coerce')
    raw_batch2and3_df['date'] = pd.to_datetime(raw_batch2and3_df['date'], errors='coerce')
    raw_batch1_df.rename(columns={'Date': 'date'}, inplace=True)
    raw_batch1_df.rename(columns={'SiteID': 'site'}, inplace=True)
    raw_batch1_df.rename(columns={'EC': 'FTIR_EC'}, inplace=True)
    raw_batch1_df['date'] = pd.to_datetime(raw_batch2and3_df['date'], errors='coerce')
    # Handle missing values or NA values as necessary
    raw_batch4_df = raw_batch4_df.fillna(0)
    raw_batch2and3_df = raw_batch2and3_df.fillna(0)
    raw_batch1_df = raw_batch1_df.fillna(0)
    # Check the columns for both dataframes and ensure consistency
    # common_columns = ['site', 'date', 'FTIR_EC', 'EC MDL']
    common_columns = ['site', 'date', 'FTIR_EC']
    raw_batch4_df = raw_batch4_df[common_columns]
    raw_batch2and3_df = raw_batch2and3_df[common_columns]
    raw_batch1_df = raw_batch1_df[common_columns]
    # Merge the dataframes
    FTIR_raw_df = pd.concat([raw_batch4_df, raw_batch2and3_df, raw_batch1_df], ignore_index=True)
    FTIR_raw_df = FTIR_raw_df.dropna(subset=['date', 'FTIR_EC'])
    FTIR_raw_df.rename(columns={'date': 'start_date'}, inplace=True)
    FTIR_raw_df.rename(columns={'site': 'Site'}, inplace=True)
    FTIR_raw_df = FTIR_raw_df[FTIR_raw_df['start_date'] != 0]
    return FTIR_raw_df

# Replace invalid entries with NaN
def fill_invalid_with_nan(HIPS_FTIR_df):
    # Specify columns to check
    columns_to_check = ['BC_HIPS_ug', 'EC_FTIR_ug_master', 'BC_HIPS_ug/m3', 'EC_FTIR_ug/m3_master']
    for col in columns_to_check:
        HIPS_FTIR_df[col] = HIPS_FTIR_df[col].replace({0: np.nan})  # Replace 0 with NaN
        HIPS_FTIR_df[col] = HIPS_FTIR_df[col].where(HIPS_FTIR_df[col].notna(), np.nan)  # Keep existing NaNs
    return HIPS_FTIR_df

# Main script
if __name__ == "__main__":
    # Read data
    HIPS_FTIR_df = read_master_files(obs_dir)
    HIPS_FTIR_df = fill_invalid_with_nan(HIPS_FTIR_df)
    UV_df = read_UV_Vis_files(out_dir)
    FTIR_raw_df = read_FTIR_raw_data(FTIR_dir)

    # Combine the HIPS and UV-Vis datasets based on FilterID
    BC_df = pd.merge(UV_df, HIPS_FTIR_df, on=['FilterID'], how='outer') #include all rows from both datasets, filling in NaN for missing matches.
    BC_df['BC_UV-Vis_ug'] = BC_df['f_BC'] * BC_df['mass_ug']
    BC_df['BC_UV-Vis_ug/m3'] = BC_df['BC_UV-Vis_ug'] / BC_df['Volume_m3']
    # Rename the Site_x column to Site
    BC_df.drop('Site_x', axis=1, inplace=True)
    BC_df.rename(columns={'Site_y': 'Site'}, inplace=True)
    # Convert year, month, and day to numeric and strip any whitespace
    BC_df['start_year'] = pd.to_numeric(BC_df['start_year'].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    BC_df['start_month'] = pd.to_numeric(BC_df['start_month'].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    BC_df['start_day'] = pd.to_numeric(BC_df['start_day'].astype(str).str.strip(), errors='coerce').fillna(0).astype(int)
    BC_df['start_date'] = pd.to_datetime(
        BC_df['start_year'].astype(str) + '-' +
        BC_df['start_month'].astype(str) + '-' +
        BC_df['start_day'].astype(str),
        errors='coerce'
    ).dt.date  # Retain only the date, discard the time part

    # Combine HIPS, UV-Vis, and FTIR datasets based on date and site
    BC_df['start_date'] = pd.to_datetime(BC_df['start_date'])
    FTIR_raw_df['start_date'] = pd.to_datetime(FTIR_raw_df['start_date'], errors='coerce')
    FTIR_raw_df = FTIR_raw_df.dropna(subset=['start_date'])
    BC_df = pd.merge(BC_df, FTIR_raw_df[['Site', 'start_date', 'FTIR_EC']], on=['Site', 'start_date'], how='left')
    BC_df.rename(columns={'FTIR_EC': 'EC_FTIR_ug/m3_raw'}, inplace=True)
    # Read site details and merge
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'),
                            usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])
    obs_df = pd.merge(BC_df, site_df, how='left', left_on='Site', right_on='Site_Code').drop('Site_Code', axis=1)
    obs_df.columns = obs_df.columns.str.strip()
    # Write HIPS data to Excel
    with pd.ExcelWriter(os.path.join(out_dir, "BC_HIPS_FTIR_UV-Vis_SPARTAN_allYears.xlsx"), engine='openpyxl', mode='w') as writer:
        obs_df.to_excel(writer, sheet_name='All', index=False)

    # obs_df = obs_df[obs_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]

    # Create summary statistics by 'Country' and 'City'
    summary_df = obs_df.groupby(['Country', 'City']).agg(
        earliest_date_HIPS=('start_date', lambda x: x[obs_df['BC_HIPS_ug/m3'].notnull()].min().date()),
        latest_date_HIPS=('start_date', lambda x: x[obs_df['BC_HIPS_ug/m3'].notnull()].max().date()),
        earliest_date_EC_FTIR_master=('start_date', lambda x: x[obs_df['EC_FTIR_ug/m3_master'].notnull()].min().date()),
        latest_date_EC_FTIR_master=('start_date', lambda x: x[obs_df['EC_FTIR_ug/m3_master'].notnull()].max().date()),
        earliest_date_EC_FTIR_raw=('start_date', lambda x: x[obs_df['EC_FTIR_ug/m3_raw'].notnull()].min().date()),
        latest_date_EC_FTIR_raw=('start_date', lambda x: x[obs_df['EC_FTIR_ug/m3_raw'].notnull()].max().date()),
        earliest_date_UV_Vis=('start_date', lambda x: x[obs_df['BC_UV-Vis_ug/m3'].notnull()].min().date()),
        latest_date_UV_Vis=('start_date', lambda x: x[obs_df['BC_UV-Vis_ug/m3'].notnull()].max().date()),
        count_BC_HIPS=('BC_HIPS_ug/m3', 'count'),
        count_EC_FTIR_master=('EC_FTIR_ug/m3_master', 'count'),
        count_EC_FTIR_raw=('EC_FTIR_ug/m3_raw', 'count'),
        count_BC_UV_Vis=('BC_UV-Vis_ug/m3', 'count')
    ).reset_index()
    with pd.ExcelWriter(os.path.join(out_dir, "BC_HIPS_FTIR_UV-Vis_SPARTAN_allYears.xlsx"), engine='openpyxl', mode='a') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
