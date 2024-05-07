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


# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/'
################################################################################################
# Match FTIR_OM and Residual
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(Residual_dir):
    Residual_dfs = []
    for filename in os.listdir(Residual_dir):
        if filename.endswith('.csv'):
            try:
                master_data = pd.read_csv(os.path.join(Residual_dir, filename), skiprows=3, encoding='ISO-8859-1')
                # print(f"First few rows of file '{filename}':")
                # print(master_data.head())
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
    OM_22_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2022_06', usecols=['Site', 'Date', 'M_Total', 'OC'])
    OM_22_df.rename(columns={'M_Total': 'OM_new'}, inplace=True)
    OM_22_df.rename(columns={'OC': 'FTIR_OC_new'}, inplace=True)
    OM_22_new_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2022_06_new',
                                 usecols=['Site', 'Date', 'OM', 'FTIR_OC'])
    # print(OM_df.head())
    # print(Residual_df.head())
    # Merge Residual and OM df based on matching values of "Site" and "Date"
    merged_df = pd.merge(OM_22_new_df, OM_22_df, on=['Site', 'Date'], how='inner')
    merged_df.rename(columns={'Country': 'country'}, inplace=True)
    merged_df.rename(columns={'City': 'city'}, inplace=True)
    # Write to Excel
    with pd.ExcelWriter(os.path.join(out_dir, 'OM_Residual_SPARTAN.xlsx'), engine='openpyxl', mode='a') as writer:
        merged_df.to_excel(writer, sheet_name='OM_Residual_22_22new', index=False)


