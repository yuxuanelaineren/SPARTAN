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
from scipy.stats import linregress
cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-EDGARv61-vert-{}-output/monthly/'.format(cres.lower(), deposition) # EDGAR, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-HTAPv3-vert-{}-output/monthly/'.format(cres.lower(), deposition) # HTAP, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-CSwinds-output/monthly/'.format(cres.lower(), deposition) # CEDS, C3720, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, LUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, GEOS-FP
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-merra2-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, MERRA2
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
otherMeas_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/'

################################################################################################
# Match all FTIR_EC and HIPS_BC
################################################################################################
# Read data
FTIR_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/'
HIPS_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='All')
HIPS_df['Date'] = pd.to_datetime(
    HIPS_df['start_year'].astype(str) + '-' + HIPS_df['start_month'].astype(str) + '-' + HIPS_df['start_day'].astype(str))
site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City'])
FTIR_b4_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch4_2024_03', usecols=['site', 'date', 'FTIR_EC'], skiprows=1)
FTIR_b2_b3_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch2_v3_batch3_v2', usecols=['site', 'date', 'FTIR_EC'], skiprows=1)
FTIR_b1_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch1_2020_08', usecols=['Site', 'Date', 'FTIR_EC'])
FTIR_b4_df.rename(columns={'FTIR_EC': 'EC', 'site': 'Site', 'date': 'Date'}, inplace=True)
FTIR_b2_b3_df.rename(columns={'FTIR_EC': 'EC', 'site': 'Site', 'date': 'Date'}, inplace=True)
# Add a 'batch' column to each DataFrame
FTIR_b4_df['batch'] = 'batch4_2024_03'
FTIR_b2_b3_df['batch'] = 'batch2_2022_06_batch3_2023_03'
FTIR_b1_df['batch'] = 'batch1_2020_08'
FTIR_df = pd.concat([FTIR_b4_df, FTIR_b2_b3_df]) # exlcude batch 1 as no lot specific calibrations
# Merge Residual and OM df based on matching values of "Site" and "Date"
merged_df = pd.merge(HIPS_df, FTIR_df, on=['Site', 'Date'], how='inner')
merged_df.rename(columns={'BC': 'BC'}, inplace=True)
# merged_df.rename(columns={'Country': 'country'}, inplace=True)
# merged_df.rename(columns={'City': 'city'}, inplace=True)
# Write to Excel
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_BC_FT-IR_EC_20250319.xlsx', engine='openpyxl', mode='w') as writer: # write mode
    merged_df.to_excel(writer, sheet_name='EC_batch234_HIPS_BC', index=False)