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
# Combine FTIR OC and extracted GCHP OM/OC based on lat/lon and seasons
################################################################################################
# Define a function to map months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'JJA'
    elif month in [6, 7, 8]:
        return 'MAM'
    elif month in [9, 10, 11]:
        return 'SON'
    else:
        return 'Unknown'

# Load data
obs_df = pd.read_excel(out_dir + 'FT-IR_OM_OC_Residual_Chris.xlsx', sheet_name='OM_OC_batch234_Residual')
sim_df = pd.read_excel(out_dir + 'sim_OMOC_at_SPARTAN_site.xlsx', sheet_name='All_Seasons')

# Add a new column 'season' based on 'Start_Month_local'
obs_df.rename(columns={'start_month': 'month'}, inplace=True)
obs_df['season'] = obs_df['month'].apply(get_season)
obs_df.rename(columns={'OM': 'FTIR_OM', 'OC': 'FTIR_OC'}, inplace=True)

# Merge obs_df and sim_df based on 'season', 'Latitude', and 'Longitude' in obs_df, and 'season', 'site_lat', and 'site_lon' in sim_df
sim_obs_df = pd.merge(obs_df, sim_df, left_on=['season', 'Country', 'City'], right_on=['season', 'Country', 'City'], how='inner')

# Drop the redundant latitude and longitude columns from site_df
# sim_obs_df.drop(columns=['sim_lat', 'sim_lon'], inplace=True)

print('sim_obs_df:', sim_obs_df)

# Write the summary DataFrame to an Excel file
with pd.ExcelWriter(out_dir + 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx', engine='openpyxl', mode='w') as writer:
    sim_obs_df.to_excel(writer, sheet_name='All', index=False)