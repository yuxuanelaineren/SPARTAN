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
from matplotlib import font_manager
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
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
# Create scatter plot for annual data
################################################################################################
# Read the file
annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_AnnualMean.xlsx'.format(cres, species, year)), sheet_name='All')

# Drop rows where BC is greater than 1
annual_df = annual_df.loc[annual_df['obs'] <= 20]

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=annual_df, hue='City', s=25, alpha=1, ax=ax)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Modify legend background color and position
legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)
# legend.get_frame().set_linewidth(0.0)  # remove legend border
legend.get_texts()[0].set_fontname("Arial")  # set fontname of the first label

# Set title, xlim, ylim, ticks, labels
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
          fontsize=14, fontname='Arial', y=1.03)  # PM$_{{2.5}}$

plt.xlim([-0.5, 16])
plt.ylim([-0.5, 16])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=14)
plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=14)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = annual_df['obs']
y = annual_df['obs']
plt.plot([annual_df['sim'].min(), annual_df['sim'].max()], [annual_df['sim'].min(), annual_df['sim'].max()],
         color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(annual_df)
plt.text(0.70, 0.3, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)
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
    plt.text(0.70, 0.35, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=14)

plt.xlabel('SPARTAN Black Carbon (µg/m$^3$)', fontsize=14, color='black', fontname='Arial')
plt.ylabel('GCHP Black Carbon (µg/m$^3$)', fontsize=14, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
plt.savefig(out_dir + '{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
plt.show()

