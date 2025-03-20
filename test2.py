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
# Extract BC_HIPS from masterfile and lon/lat from site.details
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(obs_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350', 'MXMC-0006', 'NGIL-0309'
    ]
    HIPS_dfs = []
    for filename in os.listdir(obs_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')
            HIPS_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                            'BC_HIPS_ug', 'Fe_XRF_ng', 'Flags']
            if all(col in master_data.columns for col in HIPS_columns):
                # Select the specified columns
                master_data.columns = master_data.columns.str.strip()
                HIPS_df = master_data[HIPS_columns].copy()
                # Exclude specific FilterID values
                HIPS_df = HIPS_df[~HIPS_df['FilterID'].isin(excluded_filters)]
                # Select PM2.5
                HIPS_df['Mass_type'] = pd.to_numeric(HIPS_df['Mass_type'], errors='coerce')
                HIPS_df = HIPS_df.loc[HIPS_df['Mass_type'] == 1]
                # Convert the relevant columns to numeric
                HIPS_df[['BC_HIPS_ug', 'mass_ug', 'Volume_m3', 'start_year', 'Fe_XRF_ng']] = HIPS_df[
                    ['BC_HIPS_ug', 'mass_ug', 'Volume_m3', 'start_year', 'Fe_XRF_ng']].apply(pd.to_numeric, errors='coerce')
                # Select year 2019 - 2023
                HIPS_df = HIPS_df[HIPS_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]
                # Drop rows with NaN values
                HIPS_df = HIPS_df.dropna(subset=['start_year', 'Volume_m3', 'BC_HIPS_ug'])
                HIPS_df = HIPS_df[HIPS_df['Volume_m3'] > 0]  # Exclude rows where Volume_m3 is 0
                HIPS_df = HIPS_df[HIPS_df['BC_HIPS_ug'] > 0]  # Exclude rows where HIPS_BC is 0
                # Calculate BC concentrations and fractions
                HIPS_df['BC'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['Volume_m3']
                HIPS_df['Fe'] = HIPS_df['Fe_XRF_ng']*0.001 / HIPS_df['Volume_m3']
                HIPS_df['PM25'] = HIPS_df['mass_ug'] / HIPS_df['Volume_m3']
                HIPS_df['BC_PM25'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['mass_ug']
                HIPS_df['Fe_PM25'] = HIPS_df['Fe_XRF_ng']*0.001 / HIPS_df['mass_ug']
                HIPS_df['Fe_BC'] = HIPS_df['Fe_XRF_ng']*0.001 / HIPS_df['BC_HIPS_ug']
                # Extract the site name and add as a column
                site_name = filename.split('_')[0]
                HIPS_df["Site"] = [site_name] * len(HIPS_df)
                # Append the current HIPS_df to the list
                HIPS_dfs.append(HIPS_df)
            else:
                print(f"Skipping {filename} because not all required columns are present.")
    return pd.concat(HIPS_dfs, ignore_index=True)

# Main script
if __name__ == "__main__":
    # Read data
    HIPS_df = read_master_files(obs_dir)
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])
    obs_df = pd.merge(HIPS_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)
    # Write HIPS data to Excel
    with pd.ExcelWriter(os.path.join(out_dir, "Fe_BC_HIPS_SPARTAN.xlsx"), engine='openpyxl', mode='w') as writer:
        obs_df.to_excel(writer, sheet_name='All', index=False)

    # Writ summary statistics to Excel
    site_counts = obs_df.groupby('Site')['FilterID'].count()
    for site, count in site_counts.items():
        print(f"{site}: {count} rows")
    summary_df = obs_df.groupby(['Country', 'City'])['Fe'].agg(['count', 'mean', 'median', 'std'])
    summary_df['stderr'] = summary_df['std'] / np.sqrt(summary_df['count']).pow(0.5)
    summary_df.rename(columns={'count': 'num_obs', 'mean': 'bc_mean', 'median': 'bc_median', 'std': 'bc_stdv', 'stderr': 'bc_stderr'},
        inplace=True)
    with pd.ExcelWriter(os.path.join(out_dir, "Fe_BC_HIPS_SPARTAN.xlsx"), engine='openpyxl', mode='a') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=True)
