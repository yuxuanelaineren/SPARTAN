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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
Colocation_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/Co-location/'
################################################################################################
# Calculate dust in Abu Dhabi
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'AbuDhabi_RCFM_FT-IR_master.xlsx'), sheet_name='All')
abu_dhabi_df = compr_df[compr_df['City'] == 'Abu Dhabi'].copy()
# Define the columns that need to be converted (all _XRF_ng columns)
XRF_columns = [col for col in compr_df.columns if col.endswith('_XRF_ng')]
# Perform conversion for each of these columns
for col in XRF_columns:
    # Convert mass to concentration (ng/m3 to µg/m³)
    abu_dhabi_df[f'{col.replace("_XRF_ng", "")}'] = abu_dhabi_df[col] / abu_dhabi_df['Volume_m3'] / 1000  # Divide by 1000 to convert ng to µg

# ==== Dust Equation ====
# Calcule MAL based on K, Mg, Na for each filetr
MAL_default = 0.72
CF_default = 1.14
# Calculate MAL based on K, Mg, Na for each filter
abu_dhabi_df['MAL_corrected'] = (
    (1.20 * abu_dhabi_df['K'] / abu_dhabi_df['Al']) +
    (1.66 * abu_dhabi_df['Mg'] / abu_dhabi_df['Al']) +
    (1.35 * abu_dhabi_df['Na'] / abu_dhabi_df['Al'])
) / 1.89
abu_dhabi_df['Soil_default'] = (
    (1.89 * abu_dhabi_df['Al'] * (1 + MAL_default)) +
    (2.14 * abu_dhabi_df['Si']) +
    (1.40 * abu_dhabi_df['Ca']) +
    (1.36 * abu_dhabi_df['Fe']) +
    (1.67 * abu_dhabi_df['Ti'])
) * CF_default
abu_dhabi_df['Soil_corrected'] = (
    (1.89 * abu_dhabi_df['Al'] * (1 + abu_dhabi_df['MAL_corrected'])) +
    (2.14 * abu_dhabi_df['Si']) +
    (1.40 * abu_dhabi_df['Ca']) +
    (1.36 * abu_dhabi_df['Fe']) +
    (1.67 * abu_dhabi_df['Ti'])
) * CF_default

# # ==== Al & Si Attenuation Correction ====
# abu_dhabi_df['dust_loading'] = abu_dhabi_df['Soil_default'] * abu_dhabi_df['Volume_m3'] / 3.53 # Convert µg/m³ to µg and then to µg/cm²
# abu_dhabi_df['A'] = 0.78 - 8.6e-4 * abu_dhabi_df['dust_loading'] + 4.0e-7 *abu_dhabi_df['dust_loading'] ** 2 # A = 0.78 - 8.6e-4 * dust_loading + 4.0e-7 * dust_loading ** 2 #
#
# # Correct Al & Si values
# abu_dhabi_df['Si_corrected'] = abu_dhabi_df['Si'] / abu_dhabi_df['A']
# abu_dhabi_df['Al_corrected'] = abu_dhabi_df['Al'] * 0.77 / abu_dhabi_df['A']  # 0.77 adjustment for Al calibration

# Calculate and print statistics for MAL_corrected and Soil_corrected
for col in ['MAL_corrected', 'Soil_corrected', 'Soil_default']:
    mean_val = abu_dhabi_df[col].mean()
    median_val = abu_dhabi_df[col].median()
    min_val = abu_dhabi_df[col].min()
    max_val = abu_dhabi_df[col].max()
    se_val = abu_dhabi_df[col].std() / (len(abu_dhabi_df[col]) ** 0.5)
    print(f"Statistics for {col}:\nMean: {mean_val}\nMedian: {median_val}\nMin: {min_val}\nMax: {max_val}\nSE: {se_val}\n")
# Save results to a new Excel file
# output_file = os.path.join(out_dir, 'AbuDhabi_MAL.xlsx')
# abu_dhabi_df[['FilterID', 'MAL_corrected', 'Soil_corrected', 'Soil_default',
#               'Soil', 'Al', 'Si', 'Ca', 'Fe', 'Ti', 'K', 'Mg', 'Na']].to_excel(output_file, index=False)
