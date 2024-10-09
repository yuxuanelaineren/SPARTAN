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
# # Calculate Population-weighted conc (pwm)
# annual_conc = None
# annual_pop = None
# for mon in range(1, 13):
#     # Load simulated data
#     with xr.open_dataset(sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') as sim_df:
#         conc = sim_df[species].values
#     # Load population data
#     pop_df = xr.open_dataset(support_dir + 'Regrid.PopDen.latlon.1800x3600.to.{}.conserve.2015.nc4'.format(cres.upper())).squeeze()
#     pop = pop_df['pop'].values
#
#     # # Mask out sea areas
#     # lsmask_df = xr.open_dataset(support_dir + 'Regrid.LL.1800x3600.{}.neareststod.landseamask.nc'.format(cres.upper())).squeeze()
#     # lsmask = lsmask_df['mask'].values
#     # land_mask = lsmask < 50  # < 50 represents land
#     # conc = np.where(land_mask, conc, np.nan)
#     # pop = np.where(land_mask, pop, np.nan)
#
#     conc = conc.flatten()
#     pop = pop.flatten()
#
#     if annual_conc is None:
#         annual_conc = conc
#         annual_pop = pop
#     else:
#         annual_conc += conc
# annual_conc /= 12
# annual_conc = annual_conc.squeeze()
# # Calculate Population-weighted conc (pwm)
# conc = annual_conc.flatten()
# pop = annual_pop.flatten()
# ind = np.where(~np.isnan(conc))
# N = len(conc[ind])
# pwm = np.nansum(pop[ind] * conc[ind]) / np.nansum(pop[ind]) # compute pwm, one value
# pwstd = np.sqrt(np.nansum(pop[ind] * (conc[ind] - pwm) ** 2) / ((N - 1) / N * np.nansum(pop[ind])))
# pwse = pwstd / np.sqrt(N)
# print(f"Population-weighted mean (pwm): {pwm}")
# print(f"Population-weighted std (pwstd): {pwstd}")
# print(f"Population-weighted se (pwse): {pwse}")
# # Compute global (arithmetic) mean, standard deviation, and standard error
# global_mean = np.nanmean(conc)
# global_std = np.nanstd(conc, ddof=1)  # ddof=1 for sample standard deviation
# global_se = global_std / np.sqrt(N)
# print(f"Global mean: {global_mean}")
# print(f"Global standard deviation: {global_std}")
# print(f"Global standard error: {global_se}")

# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(7, 5))
left = 0.03
bottom = 0.05
width = 0.94
height = 0.9
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 63], crs=ccrs.PlateCarree())
# ax.set_extent([115, 118, 39.3, 41.3], crs=ccrs.PlateCarree()) # Beijing
# ax.set_extent([70, 130, 20, 50], crs=ccrs.PlateCarree()) # China
# ax.set_extent([-130, -60, 15, 50], crs=ccrs.PlateCarree()) # US
# ax.set_extent([-10, 30, 40, 60], crs=ccrs.PlateCarree()) # Europe
# ax.set_extent([-15, 25, 40, 60], crs=ccrs.PlateCarree()) # Europe with cbar
# ax.set_extent([-130, -60, 25, 60], crs=ccrs.PlateCarree()) # NA

# Define the colormap
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.7, 0, 0)] # white-blure-green-yellow-orange-red
# colors = [(0, 0, 0.6), (0, 0, 1),(0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9), (0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)] # dark blue to light blue to light red to dark red

cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)
vmax = 4

# Accumulate data for each face over the year
annual_conc = None
for face in range(6):
    for mon in range(1, 13):
        print("Opening file:", sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))
        with xr.open_dataset(
            sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') as sim_df:  # CEDS
            x = sim_df.corner_lons.isel(nf=face)
            y = sim_df.corner_lats.isel(nf=face)
            conc = sim_df[species].isel(nf=face).load()
            if annual_conc is None:
                annual_conc = conc
            else:
                annual_conc = annual_conc + conc
        print("File closed.")
    # Calculate the annual average
    annual_conc /= 12
    annual_conc = annual_conc.squeeze()
    print(x.shape, y.shape, annual_conc.shape)
    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_conc, cmap=cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# # Add Beijing city border
# beijing_shapefile = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/Beijing_border_2020/Beijing-2020.shp'
# gdf = gpd.read_file(beijing_shapefile)
# gdf = gdf.to_crs(crs=ccrs.PlateCarree().proj4_init)  # Convert CRS to match map
# ax.add_geometries(gdf.geometry, ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=1)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_other_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
# Adjust SPARTAN observations
compar_notna.loc[compar_notna['source'] == 'SPARTAN', 'obs'] *= 0.6
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim
print(compar_notna['source'].unique())

# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Measurement
s2 = [120] * len(obs)  # outer ring: Simulation
markers = {'SPARTAN': 'o', 'other': 's'}
# Create scatter plot for other data points (squares)
for i, row in compar_notna.iterrows():
    source = row['source']
    if source != 'SPARTAN':  # Exclude SPARTAN data for now
        marker = markers.get(source, 'o')
        plt.scatter(x=row['lon'], y=row['lat'], c=row['obs'], s=s1[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=4)
        plt.scatter(x=row['lon'], y=row['lat'], c=row['sim'], s=s2[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=3)
# Create scatter plot for SPARTAN data points (circles)
for i, row in compar_notna.iterrows():
    source = row['source']
    if source == 'SPARTAN':  # Plot SPARTAN data
        marker = markers.get(source, 'o')
        plt.scatter(x=row['lon'], y=row['lat'], c=row['obs'], s=s1[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=4)
        plt.scatter(x=row['lon'], y=row['lat'], c=row['sim'], s=s2[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=3)

# Calculate mean and standard error for SPARTAN sites
spartan_data = compar_notna[compar_notna['source'] == 'SPARTAN']
mean_obs = np.mean(spartan_data['obs'])
std_error_obs = np.std(spartan_data['obs']) / np.sqrt(len(spartan_data['obs']))
mean_sim = np.mean(spartan_data['sim'])
std_error_sim = np.std(spartan_data['sim']) / np.sqrt(len(spartan_data['sim']))
# Calculate NMD and NMB
NMD = np.sum(np.abs(spartan_data['sim'] - spartan_data['obs'])) / np.sum(spartan_data['obs'])
NMB = np.sum(spartan_data['sim'] - spartan_data['obs']) / np.sum(spartan_data['obs'])
# Print the final values
print(f"Normalized Mean Difference (NMD): {NMD:.4f}")
print(f"Normalized Mean Bias (NMB): {NMB:.4f}")

# Add text annotations to the plot
ax.text(0.3, 0.14, f'Normalized Mean Difference (NMD) = {NMD:.2f}', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.3, 0.14, f'Meas = {mean_obs:.1f} ± {std_error_obs:.2f} µg/m$^3$', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.3, 0.08, f'Sim at Meas = {mean_sim:.1f} ± {std_error_sim:.2f} µg/m$^3$', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.3, 0.02, f'Sim (Population-weighted) = {pwm:.1f} ± {pwse:.4f} µg/m$^3$', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.92, 0.05, f'{year}', fontsize=14, fontname='Arial', transform=ax.transAxes)
# plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial') # PM$_{{2.5}}$

# Create an inset axes for the color bar at the left middle of the plot
cbar_axes = inset_axes(ax,
                           width='1.5%',
                           height='50%',
                           bbox_to_anchor=(-0.95, -0.45, 1, 1),  # (x, y, width, height) relative to top-right corner
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
cbar = plt.colorbar(im, cax=cbar_axes, orientation="vertical")
font_properties = font_manager.FontProperties(family='Arial', size=12)
cbar.set_ticks([0, 1, 2, 3, 4], fontproperties=font_properties)
cbar.ax.set_ylabel(f'{species} (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
cbar.ax.tick_params(axis='y', labelsize=12)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

# plt.savefig(out_dir + 'Fig2_WorldMap_{}_{}_{}_Sim_vs_SPARTAN_other_{}_{}_AnnualMean_MAC10_PWM.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()
