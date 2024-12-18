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
deposition = 'LUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(
    cres.lower(), deposition)  # CEDS, C360, noLUO
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
# Create scatter plot: noLUO vs LUO, colored by major Sim vs Meas plot (blue and red)
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir + 'C360_CEDS_LUO_vs_C360_CEDS_noLUO_201907.xlsx'))
# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")
# Define city-to-color and marker mappings
city_legend = {
    'Fajardo': {'color': '#b2c2e6', 'marker': 'd'},
    'Halifax': {'color': '#91ace4', 'marker': 'd'},
    'Sherbrooke': {'color': '#6e95ff', 'marker': 'd'},
    'Melbourne': {'color': '#6a8af3', 'marker': '*'},
    'Pasadena': {'color': '#6680e6', 'marker': 'd'},
    'Ulsan': {'color': '#325fcf', 'marker': '^'},
    'Taipei': {'color': '#0044cc', 'marker': '^'},
    'Haifa': {'color': '#0021e6', 'marker': 'p'},
    'Rehovot': {'color': '#0000ff', 'marker': 'p'},
    'Seoul': {'color': '#0000cc', 'marker': '^'},
    'Kaohsiung': {'color': '#000099', 'marker': '^'},
    'Beijing': {'color': '#e68080', 'marker': '^'},
    'Mexico City': {'color': '#ef8686', 'marker': 'd'},
    'Pretoria': {'color': '#fa7070', 'marker': 'o'},
    'Johannesburg': {'color': '#ff5252', 'marker': 'o'},
    'Abu Dhabi': {'color': '#ff2929', 'marker': 'p'},
    'Ilorin': {'color': '#fd0000', 'marker': 'o'},
    'Bandung': {'color': '#eb0000', 'marker': 's'},
    'Bujumbura': {'color': '#d60000', 'marker': 'o'},
    'Kanpur': {'color': '#bc0000', 'marker': 's'},
    'Addis Ababa': {'color': '#9d0000', 'marker': 'o'},
    'Dhaka': {'color': '#800000', 'marker': 's'},
}

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Add 1:1 line with grey dash
plt.plot([-0.5, 22], [-0.5, 22], color='grey', linestyle='--', linewidth=1, zorder=1)

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(
    x='noLUO',
    y='LUO',
    data=compr_df,
    hue='city',
    palette={city: city_legend[city]['color'] for city in city_legend},
    style='city',
    markers={city: city_legend[city]['marker'] for city in city_legend},
    s=80,
    alpha=1,
    edgecolor='k'
)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Customize legend to match the order and appearance
handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(city)] for city in city_legend if city in labels]
ordered_labels = [city for city in city_legend if city in labels]
legend = plt.legend(ordered_handles, ordered_labels, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 7])
plt.ylim([-0.5, 7])
plt.xticks([0, 2, 4, 6], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6], fontname='Arial', size=18)
# plt.xlim([-0.5, 17])
# plt.ylim([-0.5, 17])
# plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
# plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Define the range of x-values for the two segments
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['noLUO'][mask], compr_df['LUO'][mask])
# Plot regression lines
sns.regplot(x='noLUO', y='LUO', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
# plt.text(0.65, 0.03, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)
plt.text(0.75, 0.03, f'July, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('BC with Default Scavenging (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('BC with Alternative Scavenging (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
plt.savefig(out_dir + 'FigS4_Scatter_C360_CEDS_LUO_vs_C360_CEDS_noLUO_201907_BlueRed.svg', dpi=300)
plt.show()