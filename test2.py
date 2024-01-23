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

cres = 'C360'
year = 2018
species = 'BC'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower())
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}/'.format(cres.lower(), species)
################################################################################################
# Create scatter plot for monthly and annual data
################################################################################################
# Read the file
# annual_df = pd.read_csv(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}01_MonMean.csv'.format(cres, species, year)))
# annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, species, year)), sheet_name='Annual')
annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_CAWNET_{}_{}_Summary.xlsx'.format(cres, species, year)), sheet_name='Annual')

# Drop rows where BC is greater than 1
# annual_df = annual_df.loc[annual_df['obs'] <= 20]

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=annual_df, hue='city', s=80, alpha=1, ax=ax, edgecolor='k')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)

# Set title, xlim, ylim, ticks, labels
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
          fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 12])
plt.ylim([-0.5, 12])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([0, 4, 8, 12], fontname='Arial', size=18)
plt.yticks([0, 4, 8, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = annual_df['obs']
y = annual_df['obs']
plt.plot([annual_df['obs'].min(), annual_df['obs'].max()], [annual_df['obs'].min(), annual_df['obs'].max()],
         color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(annual_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=22)
plt.text(0.85, 0.05, f'2018', transform=scatterplot.transAxes, fontsize=18)
# plt.text(0.05, 0.81, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)

# Perform linear regression with NaN handling
mask = ~np.isnan(annual_df['obs']) & ~np.isnan(annual_df['sim'])
slope, intercept, r_value, p_value, std_err = stats.linregress(annual_df['obs'][mask], annual_df['sim'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='obs', y='sim', data=annual_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=22)

plt.xlabel('CAWNET BC/(PM$_{{2.5}}$ - Nitrate)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('GCHP BC/(PM$_{{2.5}}$ - Nitrate)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
plt.savefig('/Users/renyuxuan/Downloads/' + 'Scatter_{}_Sim_vs_CAWNET_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)

plt.show()
