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
# Create scatter plot for Residual vs OM, colored by region
################################################################################################
# Read the file
compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_BC_FT-IR_EC_20250319.xlsx', sheet_name='EC_batch234_HIPS_BC')
compr_df['HIPS'] = compr_df['BC']
compr_df['FT-IR'] = compr_df['EC']
compr_df = compr_df[compr_df['City'].isin(['Dhaka', 'Addis Ababa'])]
# Define custom color mapping
color_map = {'Dhaka': 'red', 'Addis Ababa': 'blue'}

# Calculate mean and standard error grouped by City
summary_stats = compr_df.groupby('City').agg(
    OM_mean=('HIPS', 'mean'),
    OM_se=('HIPS', lambda x: x.std() / np.sqrt(len(x))),
    Residual_mean=('FT-IR', 'mean'),
    Residual_se=('FT-IR', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 7))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='HIPS', y='FT-IR', data=compr_df, hue='City', palette=color_map, s=60, alpha=1, edgecolor='k', marker='o')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Create legend with custom handles
legend = plt.legend(facecolor='white', bbox_to_anchor=(0.75, 0.05), loc='lower left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('HIPS vs FT-IR', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [compr_df['HIPS'].min(), compr_df['HIPS'].max()]
# Perform linear regression for all segments
mask = (compr_df['HIPS'] >= x_range[0]) & (compr_df['HIPS'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['HIPS'][mask], compr_df['FT-IR'][mask])
# # Plot regression lines
# sns.regplot(x='OM', y='Residual', data=compr_df[mask],
#             scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add columns for normalized mean difference (NMD) and normalized root mean square difference (NRMSD)
def calculate_nmd_and_nrmsd(df, obs_col, sim_col):
    """
    Calculate normalized mean difference (NMD) and normalized root mean square difference (NRMSD).

    Args:
        df (pd.DataFrame): DataFrame containing observation and simulation columns.
        obs_col (str): Column name for observations.
        sim_col (str): Column name for simulations.

    Returns:
        dict: Dictionary containing NMD and NRMSD values.
    """
    obs = df[obs_col].values
    sim = df[sim_col].values
    # Remove rows with NaN values
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]
    # Check if there are valid data points
    if len(obs) == 0:
        return {'NMD (%)': np.nan, 'NRMSD (%)': np.nan}
    # Calculate NMD
    # nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    nmd = np.sum(sim - obs) / np.sum(obs) * 100
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMB (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='HIPS', sim_col='FT-IR')
nmd = nmd_nrmsd_results['NMB (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.70, f'y = {slope:.1f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}\nNMB = {nmd:.1f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=16, color='black')

# Set labels
plt.xlabel('HIPS BC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('FT-IR EC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_vs_FT-IR_Dhaka_Addis_updated.svg', dpi=300)

plt.show()