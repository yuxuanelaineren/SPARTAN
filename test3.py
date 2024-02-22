#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib import font_manager
import seaborn as sns
from scipy import stats

# Set the directory path
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/AMS/'
################################################################################################
# Create a box plot for NO3 concentration in extraction
################################################################################################
# Set the directory path
obs_df = pd.read_excel(out_dir + 'IC_SPARTAN.xlsx', sheet_name='All')

# Set font properties
sns.set(font='Arial')

# Create the box plot
plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='City', y='NO3', data=obs_df, color='white',
                 boxprops=dict(edgecolor='black', linewidth=1, facecolor='white'),  # Box properties
                 flierprops=dict(marker='o', markeredgecolor='black', markerfacecolor='black', markersize=4))  # Flier properties
ax.set_facecolor('white')

# Iterate over each line element to set border color and width
for line in ax.lines:
    line.set_color('black')  # set border color to black
    line.set_linewidth(1)  # set border width to 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)
ax.grid(False)  # remove the grid
ax.tick_params(axis='both', direction='inout', length=6, width=1, colors='black')

# Set labels and title
plt.xlabel('City', fontsize=18)
plt.ylabel('Nitrate Concentration (µg/mL)', fontsize=18)
plt.title('Nitrate Concentration in Extraction Solution', fontsize=20)
plt.ylim([-0.5, 36])
plt.yticks(fontname='Arial', size=18)
plt.xticks(rotation=45, ha='right', size=16)

# Show plot
plt.tight_layout()
plt.savefig(out_dir + 'Box_NO3_Extraction_Solutoin.tiff', dpi=600)
plt.show()