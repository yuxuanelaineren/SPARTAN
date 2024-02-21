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
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)

################################################################################################
# Map SPARTAN and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(12, 5))
left = 0.03  # Adjust the left position
bottom = 0.01  # Adjust the bottom position
width = 0.94  # Adjust the width
height = 0.9  # Adjust the height
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

# Define the colormap
cmap = WhGrYlRd
custom_cmap = cmap # Blue to red

# Define colormap (from white to dark red through yellow and orange)
colors = ['#f7f7f7',   # light gray
          '#ffff00',   # yellow
          '#ffA500',   # orange
          '#ff4500',   # red-orange
          '#ff0000',   # red
          '#8b0000',   # dark red
          '#4d0000']   # even darker red

# Create a LinearSegmentedColormap
cmap_name = 'custom_heat'
n_bins = 100  # Increase for smoother transition
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

vmax = 8  # 8 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

# Accumulate data for each face over the year
annual_v = None

for face in range(6):
    for mon in range(1, 13):
        sim_df = xr.open_dataset(
            sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon),
            engine='netcdf4') # CEDS

        sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
        sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']
        sim_df['BC_PM25_NO3'] = sim_df['BC'] / (sim_df['PM25'] - sim_df['NIT'])

        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        if annual_v is None:
            annual_v = v
        else:
            annual_v = annual_v + v

    # Calculate the annual average
    annual_v /= 12
    annual_v = annual_v.squeeze()
    print(x.shape, y.shape, annual_v.shape)

    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_v, cmap=custom_cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim

# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Observation
s2 = [120] * len(obs)  # outer ring: Simulation

# Create scatter plot
im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=custom_cmap, edgecolor='black',
                linewidth=1, vmin=0, vmax=vmax, zorder=4)
im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=custom_cmap, edgecolor='black',
                linewidth=1, vmin=0, vmax=vmax, zorder=3)

# Calculate the global mean of simulated and observed data
global_mean_sim = np.nanmean(sim)
global_mean_obs = np.nanmean(obs)
global_std_sim = np.nanstd(sim)
global_std_obs = np.nanstd(obs)

# Display statistics as text annotations on the plot
month_str = calendar.month_name[mon]
ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.2f}',
        fontsize=16, fontname='Arial', transform=ax.transAxes)
ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.2f}',
        fontsize=16, fontname='Arial', transform=ax.transAxes)
ax.text(0.9, 0.05, f'2019', fontsize=16, fontname='Arial', transform=ax.transAxes)
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN',
            fontsize=16, fontname='Arial') # PM$_{{2.5}}$

# Create an inset axes for the color bar at the left middle of the plot
colorbar_axes = inset_axes(ax,
                           width="2%",
                           height="60%",
                           bbox_to_anchor=(-0.95, -0.35, 1, 1),  # (x, y, width, height) relative to top-right corner
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
cbar = plt.colorbar(im, cax=colorbar_axes, orientation="vertical")
num_ticks = 5
cbar.locator = plt.MaxNLocator(num_ticks)
cbar.update_ticks()
font_properties = font_manager.FontProperties(family='Arial', size=14)
cbar.set_label(f'{species} (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
cbar.ax.tick_params(axis='y', labelsize=14)

plt.savefig(out_dir + 'Fig2_{}_{}_{}_Sim_vs_SPARTAN_{}_{}_AnnualMean.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()
