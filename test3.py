#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
import calendar
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

# Create box plots
plt.figure(figsize=(12, 8))
ax = sns.boxplot(data=obs_df[['F_ug/ml', 'Cl_ug/ml', 'NO2_ug/ml', 'Br_ug/ml', 'NO3_ug/ml', 'PO4_ug/ml', 'SO4_ug/ml', 'Anion_ug/ml',
                         'Li_ug/ml', 'Na_ug/ml', 'NH4_ug/ml', 'K_ug/ml', 'Mg_ug/ml', 'Ca_ug/ml', 'Cation_ug/ml']],
            color='white', boxprops=dict(edgecolor='black', linewidth=1, facecolor='white'),  # Box properties
            flierprops=dict(marker='o', markeredgecolor='black', markerfacecolor='black', markersize=4))

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
plt.ylabel('Concentration (µg/mL)', fontsize=18)
plt.title('Concentration of Ions in Extraction Solution', fontsize=20)
plt.ylim([-0.5, 15])
plt.yticks([0, 3, 6, 9, 12, 15], fontname='Arial', size=18)
plt.yticks(fontname='Arial', size=18)
ax.set_xticklabels(['F', 'Cl', 'NO$_2^-$', 'Br', 'NO$_3^-$', 'PO$_4^{2-}$', 'SO$_4^{2-}$', 'Anion', 'Li', 'Na', 'NH$_4^+$', 'K', 'Mg', 'Ca', 'Cation'],
                   fontname='Arial', size=18)

plt.tight_layout()
plt.savefig(out_dir + 'Box_All_Extraction_Solution.tiff', dpi=600)
plt.show()
