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
import matplotlib.patches as patches
import matplotlib.lines as mlines
cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}-{}/monthly/'.format(cres.lower(), deposition) # CEDS, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower()) # HTAP, LUO
# sim_dir = '/Volumes/rvmartin/Active/dandan.z/AnalData/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/WUCR3-C360/' # EDGAR, LUO
# sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year) # C720, HTAP, LUO
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
################################################################################################
# Other: Create scatter plot, all black
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_other_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
# Convert from MAC=6 to MAC=10 in HIPS BC
compr_df.loc[compr_df['source'] == 'SPARTAN', 'obs'] *= 0.6
# compr_df = compr_df[compr_df['country'].isin(['US', 'Canada'])] # 'Europe', 'US', 'Canada'

# # Print the names of each city
# unique_cities = compr_df['city'].unique()
# for city in unique_cities:
#     print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [0, 1.4]
x_range_2 = [1.4, 11]
x_range = [0, 11]
# # Assign colors based on the x-range
# def assign_color(value):
#     if x_range_1[0] <= value <= x_range_1[1]:
#         return 'blue'
#     elif x_range_2[0] < value <= x_range_2[1]:
#         return 'red'
#     return 'black'
# compr_df['color'] = compr_df['obs'].apply(assign_color)

# Assign colors based on the 'region' column
def assign_color(region_value):
    if region_value == 'Global South':
        return 'red'
    else:
        return 'blue'
compr_df['color'] = compr_df['region'].apply(assign_color)
# Print the unique contents of the 'region' column
unique_regions = compr_df['region'].unique()
print("Unique regions in the dataset:", unique_regions)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(7, 6))
# Create scatter plot with different markers for SPARTAN and other
markers = {'SPARTAN': 'o', 'other': 's'}
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, s=60, alpha=0.9, style='source', markers=markers, hue='color', palette=[(0, 0.2, 0.9), (0.7, 0, 0)], edgecolor='k')
sns.set(font='Arial')
# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
# plt.xlim([-0.1, 1.5])
# plt.ylim([-0.1, 1.5])
# # Set tick locations and labels manually
# plt.xticks([0, 0.5, 1, 1.5], ['0', '0.5', '1', '1.5'], fontname='Arial', size=18)
# plt.yticks([0, 0.5, 1, 1.5], ['0', '0.5', '1', '1.5'], fontname='Arial', size=18)
plt.xlim([-0.5, 11])
plt.ylim([-0.5, 11])
# Set tick locations and labels manually
plt.xticks([0, 2, 4, 6, 8, 10], ['0', '2', '4', '6', '8', '10'], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], ['0', '2', '4', '6', '8', '10'], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
plt.plot([compr_df['obs'].min(), compr_df['obs'].max()], [compr_df['obs'].min(), compr_df['obs'].max()], color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['obs'][mask], compr_df['sim'][mask])
# Plot regression lines
sns.regplot(x='obs', y='sim', data=compr_df[mask], scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.6, 0.85, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.6, 0.79, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')

# Set labels
plt.xlabel('Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create a custom legend for the source only

spartan_legend = mlines.Line2D([], [], color=(0, 0.2, 0.9), markeredgecolor='k', marker='o', linestyle='None', markersize=8, label='SPARTAN')
other_legend = mlines.Line2D([], [], color=(0, 0.2, 0.9), markeredgecolor='k', marker='s', linestyle='None', markersize=8, label='other Meas')
legend = plt.legend(handles=[spartan_legend, other_legend], fontsize=12, frameon=False, loc=(0.75, 0.05))
plt.setp(legend.get_texts(), fontname='Arial')
legend.get_frame().set_facecolor('white') # Disable Legend
# plt.legend().remove()

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS6_Scatter_{}_{}_{}_Sim_vs_SPARTAN_other_{}_AnnualMean_MAC10_ColorByRegion.svg'.format(cres, inventory, deposition, species), dpi=300)

plt.show()
