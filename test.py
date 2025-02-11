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
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
################################################################################################
# Match FTIR_OM, FTIR_OC, and Residual
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(Residual_dir):
    Residual_dfs = []
    for filename in os.listdir(Residual_dir):
        if filename.endswith('.csv'):
            try:
                master_data = pd.read_csv(os.path.join(Residual_dir, filename), skiprows=3, encoding='ISO-8859-1',
                                          usecols=['Site_Code', 'Latitude', 'Longitude', 'Start_Year_local','Start_Month_local',
                                                   'Start_Day_local', 'Parameter_Name', 'Value', 'Flag'])
                # Select Residual
                Residual_df = master_data.loc[master_data['Parameter_Name'] == 'Residual Matter'].copy()
                Residual_df.rename(columns={'Site_Code': 'Site'}, inplace=True)
                # Combine date
                Residual_df['Date'] = pd.to_datetime(Residual_df['Start_Year_local'].astype(str) + '-' + Residual_df['Start_Month_local'].astype(str) + '-' + Residual_df['Start_Day_local'].astype(str))
                # Append the current HIPS_df to the list
                Residual_dfs.append(Residual_df)
            except Exception as e:
                print(f"Error occurred while processing file '{filename}': {e}. Skipping to the next file.")
    return pd.concat(Residual_dfs, ignore_index=True)

# Main script
if __name__ == '__main__':
    # Read data
    Residual_df = read_master_files(Residual_dir)
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City'])
    Residual_df = pd.merge(Residual_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)
    OM_23_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2023_03', usecols=['Site', 'Date', 'OM', 'FTIR_OC'])
    OM_22_new_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2022_06_new', usecols=['Site', 'Date', 'OM', 'FTIR_OC'])
    OM_20_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2020_09', usecols=['Site', 'Date', 'OM', 'OC'])
    OM_23_df.rename(columns={'FTIR_OC': 'OC'}, inplace=True)
    OM_22_new_df.rename(columns={'FTIR_OC': 'OC'}, inplace=True)
    # Add a 'batch' column to each DataFrame
    OM_23_df['batch'] = '2023_03'
    OM_22_new_df['batch'] = '2022_06_new'
    OM_20_df['batch'] = '2020_09'
    OM_df = pd.concat([OM_20_df, OM_22_new_df, OM_23_df])
    # Merge Residual and OM df based on matching values of "Site" and "Date"
    merged_df = pd.merge(Residual_df, OM_df, on=['Site', 'Date'], how='inner')
    merged_df.rename(columns={'Value': 'Residual'}, inplace=True)
    # merged_df.rename(columns={'Country': 'country'}, inplace=True)
    # merged_df.rename(columns={'City': 'city'}, inplace=True)
    # Write to Excel
    with pd.ExcelWriter(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_SPARTAN.xlsx'), engine='openpyxl', mode='w') as writer: # write mode
        merged_df.to_excel(writer, sheet_name='OM_OC_20_22new_23_Residual', index=False)