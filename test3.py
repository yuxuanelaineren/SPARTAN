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
# Create stack bar plot and pir charts for Abu Dhabi
################################################################################################
# Read the file
file_path = os.path.join(out_dir, 'AbuDhabi_RCFM_FT-IR_master.xlsx')
compr_df = pd.read_excel(file_path)
abu_dhabi_df = compr_df[compr_df['City'] == 'Abu Dhabi'].copy()
# Define the columns that need to be converted (all _XRF_ng columns)
XRF_columns = [col for col in compr_df.columns if col.endswith('_XRF_ng')]
# Perform conversion for each of these columns
for col in XRF_columns:
    # Convert mass to concentration (ng/m3 to µg/m³)
    abu_dhabi_df[f'{col.replace("_XRF_ng", "")}'] = abu_dhabi_df[col] / abu_dhabi_df['Volume_m3'] / 1000  # Divide by 1000 to convert ng to µg
# abu_dhabi_df['Soil × 0.1'] = abu_dhabi_df['Soil'] * 0.1
# compr_df['TEO × 0.05'] = compr_df['TEO'] * 0.05
# compr_df['Zn × 0.1'] = compr_df['Zn'] * 0.1
# compr_df['Pb × 0.1'] = compr_df['Pb'] * 0.1
# compr_df['Si × 0.1'] = compr_df['Si'] * 0.1

# Select species for the stacked bar plot
columns_to_plot = ['Al', 'Si', 'Ca', 'Fe', 'Ti']
abu_dhabi_df[columns_to_plot] = abu_dhabi_df[columns_to_plot].clip(lower=0)
# Define species colors for plotting
species_colors = {
    'Al': '#4D4D4D',       # Dark Gray
    'Si': '#C2B280',       # Sandy Brown
    'Ca': '#D9D9D9',       # Light Gray
    'Fe': '#B7410E',       # Rusty Brown
    'Ti': '#A0C4FF',       # Light Blue
    # 'Soil': '#B8860B',     # Earthy Brown
}

# Convert dates to numerical format for better width control
dates_num = mdates.date2num(abu_dhabi_df['Date'])
# Define bar width and offset for Soil bars
bar_width = 7  # Reduce width for better spacing
offset = 4  # Shift soil bars slightly to the right
# Initialize bottom values for stacking
bottom_values = np.zeros(len(abu_dhabi_df))
# Convert dates to numerical format for better width control
dates_num = mdates.date2num(abu_dhabi_df['Date'])

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
# # Add a grey dashed line at y = 0
# ax.axhline(0, color='grey', linestyle='dashed', linewidth=1)
# Loop through species and plot in order (stacked bars)
for specie in columns_to_plot:
    if specie in species_colors:
        ax.bar(dates_num, abu_dhabi_df[specie], width=bar_width, label=specie,
               color=species_colors[specie], bottom=bottom_values, align='center')
        bottom_values += abu_dhabi_df[specie]  # Accumulate values for stacking

# # Plot Soil as a separate bar next to each stacked bar
# if 'Soil × 0.1' in abu_dhabi_df.columns:
#     ax.bar(dates_num + offset, abu_dhabi_df['Soil'], width=bar_width, label='Soil',
#            color='yellow', edgecolor='white', align='center')

# Format x-axis
ax.set_xlabel('Date', fontsize=16, family='Arial')
ax.set_ylabel('Concentration (µg/m$^3$)', fontsize=16, family='Arial')
ax.set_title('Stacked Bar Plot of Dust Components: Abu Dhabi', fontsize=18, family='Arial')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format x-axis labels
plt.xticks(rotation=45, fontsize=12, family='Arial')
plt.yticks([0, 4, 8, 12, 16], fontsize=12, family='Arial')
plt.ylim([0, 16])

# Add legend
legend = ax.legend(prop={'family': 'Arial', 'size': 12}, bbox_to_anchor=(1.01, 1), loc='upper left')
legend.get_frame().set_edgecolor('black')
# Set font for all texts in the plot
# plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})
plt.tight_layout()
# Save the plot as a JPG with 300 DPI
output_path = os.path.join(out_dir, 'Stacked_Bar_AbuDhabi_Dust.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='jpg')

# Show plot
plt.show()
# # Filter data for Abu Dhabi
# abu_dhabi_df = compr_df[compr_df['City'] == 'Abu Dhabi']
# # Select the relevant columns for averaging
# columns_to_average = ['BC', 'TEO', 'Soil', 'Na', 'NO3', 'NH4', 'SO4', 'PBW', 'Cl', 'K', 'RM_dry']
# # Compute the average values
# average_values = abu_dhabi_df[columns_to_average].mean()
# # Remove negative values
# average_values = average_values[average_values > 0]
# # Check if there are valid values left
# if not average_values.empty:
#     # Get colors for available species
#     colors = [species_colors[specie] for specie in average_values.index]
#
#     # Plot the pie chart
#     plt.figure(figsize=(8, 8))
#     plt.pie(average_values, labels=average_values.index, autopct='%1.1f%%', startangle=140,
#             colors=colors, wedgeprops={'edgecolor': 'black'})
#
#     # Set title
#     plt.title('Average Composition in Abu Dhabi')
#
#     # Show the plot
#     plt.show()
# else:
#     print("No valid positive values to plot.")