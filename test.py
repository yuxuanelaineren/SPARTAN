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
# Other Measurements: Map SPARTAN, others, and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(10, 5))
left = 0.03
bottom = 0.05
width = 0.94
height = 0.9
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 63], crs=ccrs.PlateCarree())
# ax.set_extent([70, 130, 20, 50], crs=ccrs.PlateCarree()) # China
# ax.set_extent([-130, -60, 15, 50], crs=ccrs.PlateCarree()) # US
# ax.set_extent([-10, 30, 40, 60], crs=ccrs.PlateCarree()) # Europe
# ax.set_extent([-15, 25, 40, 60], crs=ccrs.PlateCarree()) # Europe with cbar
# ax.set_extent([-130, -60, 25, 60], crs=ccrs.PlateCarree()) # NA

# Define the colormap
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.7, 0, 0)]
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)
vmax = 4

# Load the Data
pop_df = xr.open_dataset(support_dir + 'Regrid.PopDen.latlon.1800x3600.to.{}.conserve.2015.nc4'.format(cres.upper())).squeeze() # Squeeze to remove single-dimensional entries
pop = pop_df['pop'].values
lsmask_df = xr.open_dataset(support_dir + 'Regrid.LL.1800x3600.{}.neareststod.landseamask.nc'.format(cres.upper())).squeeze()
lsmask = lsmask_df['mask'].values

# Accumulate data for each face over the year
annual_v = None
for face in range(6):
    for mon in range(1, 2):
        print("Opening file:", sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))
        with xr.open_dataset(
            sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') as sim_df:  # CEDS
            x = sim_df.corner_lons.isel(nf=face)
            y = sim_df.corner_lats.isel(nf=face)
            v = sim_df[species].isel(nf=face).load()
            # # Mask out sea areas
            # land_mask = lsmask < 50  # < 50 represents land
            # v = np.where(land_mask, v, np.nan)
            # pop = np.where(land_mask, pop, np.nan)
            if annual_v is None:
                annual_v = v
            else:
                annual_v = annual_v + v
        print("File closed.")
    # Calculate the annual average
    annual_v /= 12
    annual_v = annual_v.squeeze()
    print(x.shape, y.shape, annual_v.shape)
    # Plot the annual average data for each face
    # im = ax.pcolormesh(x, y, annual_v, cmap=cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

    # Population-weighted conc (pwm)
    ind = np.where(~np.isnan(annual_v))
    N = len(annual_v[ind])
    pwm = np.nansum(pop[ind] * annual_v[ind]) / np.nansum(pop[ind]) # compute pwm, one value
    pwstd = np.sqrt(np.nansum(pop[ind] * (annual_v[ind] - pwm) ** 2) / ((N - 1) / N * np.nansum(pop[ind])))
    pwse = pwstd / np.sqrt(N)
    print(f"Population-weighted mean (pwm): {pwm}")
    print(f"Population-weighted std (pwstd): {pwstd}")
    print(f"Population-weighted se (pwse): {pwse}")