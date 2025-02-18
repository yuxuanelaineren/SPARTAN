#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
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

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/'
SPARTAN_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
################################################################################################
# Process RCFM from Chris and apppend date to filter
################################################################################################
# Function to read and preprocess RM data from Chris
def read_and_preprocess_rm_data(out_dir, filename='Updated_RCFM_Chris_raw.xlsx'):
    """
    Function to read and preprocess RM data from Chris.

    Args:
    - out_dir (str): Directory path where the Excel file is located.
    - filename (str): Name of the Excel file to read (default is 'Updated_RCFM_Chris.xlsx').

    Returns:
    - RM_df (DataFrame): A concatenated DataFrame with data from all sheets and a 'SheetName' column.
    """
    RM_df = pd.DataFrame()
    RM_data = pd.read_excel(os.path.join(out_dir, filename), sheet_name=None)
    sheet_names = RM_data.keys()
    # Loop through each sheet and append to RM_df with a new column for sheet name
    for sheet in sheet_names:
        # Read the sheet into a dataframe
        df_sheet = RM_data[sheet]
        df_sheet['Site'] = sheet  # Add a 'SheetName' column
        RM_df = pd.concat([RM_df, df_sheet], ignore_index=True)
    return RM_df
# Function to read and preprocess master data
def read_master_files(SPARTAN_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350', 'MXMC-0006', 'NGIL-0309'
    ]
    SPARTAN_dfs = []
    for filename in os.listdir(SPARTAN_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(SPARTAN_dir, filename), encoding='ISO-8859-1')
            SPARTAN_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'mass_ug', 'Volume_m3']
            if all(col in master_data.columns for col in SPARTAN_columns):
                master_data = master_data[SPARTAN_columns]
                master_data['mass_ug'] = pd.to_numeric(master_data['mass_ug'], errors='coerce')
                master_data['Volume_m3'] = pd.to_numeric(master_data['Volume_m3'], errors='coerce')
                master_data['PM2.5_master'] = master_data['mass_ug'] / master_data['Volume_m3']
                # Convert to string and remove whitespace
                master_data['start_year'] = master_data['start_year'].astype(str).str.strip()
                # Convert columns to numeric, filling invalid values with NaN, and then replace NaNs with 0 or a valid default
                master_data['start_year'] = pd.to_numeric(master_data['start_year'], errors='coerce', downcast='integer')
                # master_data['Date'] = pd.to_datetime(master_data['start_year'].astype(str) + '-' + master_data['start_month'].astype(str) + '-' + master_data['start_day'].astype(str))
                SPARTAN_dfs.append(master_data)
            else:
                print(f"Skipping {filename} because not all required columns are present.")
    return pd.concat(SPARTAN_dfs, ignore_index=True)

# Main script
if __name__ == '__main__':
    # Read RM data from Chris
    RM_df = read_and_preprocess_rm_data(out_dir)
    SPARTAN_df = read_master_files(SPARTAN_dir)
    RM_df = RM_df.merge(SPARTAN_df[['FilterID', 'start_year', 'start_month', 'start_day', 'PM2.5_master']], on='FilterID', how='left')
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City'])
    RM_df = pd.merge(RM_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)
    with pd.ExcelWriter(os.path.join(out_dir, "Updated_RCFM_Chris_Summary.xlsx"), engine='openpyxl', mode='w') as writer:
        RM_df.to_excel(writer, sheet_name='All', index=False)
    # Read FT-IR OM
    OM_b4_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch4_2024_03',
                             usecols=['site', 'date', 'OM', 'FTIR_OC'], skiprows=1)
    OM_b2_b3_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch2_v3_batch3_v2',
                                usecols=['site', 'date', 'OM', 'FTIR_OC'], skiprows=1)
    OM_b1_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch1_2020_08',
                             usecols=['Site', 'Date', 'aCOH', 'aCH', 'naCO', 'COOH', 'FTIR_OC'])
    OM_b4_df.rename(columns={'FTIR_OC': 'OC', 'site': 'Site', 'date': 'Date'}, inplace=True)
    OM_b2_b3_df.rename(columns={'FTIR_OC': 'OC', 'site': 'Site', 'date': 'Date'}, inplace=True)
    # Add a 'batch' column to each DataFrame
    OM_b4_df['batch'] = 'batch4_2024_03'
    OM_b2_b3_df['batch'] = 'batch2_2022_06_batch3_2023_03'
    OM_b1_df['batch'] = 'batch1_2020_08'
    OM_df = pd.concat([OM_b4_df, OM_b2_b3_df])  # exlcude batch 1 as no lot specific calibrations
    # Merge RM and OM df based on matching values of "Site" and "Date"
    RM_df['start_year'] = RM_df['start_year'].astype(int)
    RM_df['start_month'] = RM_df['start_month'].astype(int)
    RM_df['start_day'] = RM_df['start_day'].astype(int)
    RM_df['Date'] = pd.to_datetime(RM_df['start_year'].astype(str) + '-' +RM_df['start_month'].astype(str) + '-' + RM_df['start_day'].astype(str))

    merged_df = pd.merge(RM_df, OM_df, on=['Site', 'Date'], how='inner')
    # merged_df.rename(columns={'Country': 'country'}, inplace=True)
    # merged_df.rename(columns={'City': 'city'}, inplace=True)
    # Write to Excel
    with pd.ExcelWriter(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_Chris.xlsx'), engine='openpyxl',
                        mode='w') as writer:  # write mode
        merged_df.to_excel(writer, sheet_name='OM_OC_batch234_Residual', index=False)