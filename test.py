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

cres = 'C360'
year = 2018
species = 'BC'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower())
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}/'.format(cres.lower(), species)
################################################################################################
# Map SPARTAN and GCHP data
################################################################################################
# Map SPARTAN and GCHP data
for mon in range(1, 2):
    # Plot map using simulation data
    sim_df = xr.open_dataset(f'{sim_dir}{cres}.LUO.PM25.RH35.NOx.O3.{year}{mon:02d}.MonMean.nc4')

    plt.style.use('default')
    plt.figure(figsize=(10, 5))
    left = 0.1  # Adjust the left position
    bottom = 0.1  # Adjust the bottom position
    width = 0.8  # Adjust the width
    height = 0.8  # Adjust the height
    ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
    ax.coastlines(color=(0.4, 0.4, 0.4))  # Set the color of coastlines to dark grey
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))  # Set the color of borders to dark grey
    ax.set_global()

    ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

    # Define the colormap
    cmap = WhGrYlRd
    cmap_reversed = cmap

    # Plot data for each face
    for face in range(6):
        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        im = ax.pcolormesh(x, y, v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=100)

    # Read comparison data
    compar_filename = f'{cres}_LUO_Sim_vs_SPARTAN_{species}_{year}{mon:02d}_MonMean.csv'
    compar_df = pd.read_csv(os.path.join(out_dir, compar_filename))
    compar_notna = compar_df[compar_df.notna().all(axis=1)]
    lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim

    # Define marker sizes
    s1 = [20] * len(obs)  # inner circle: Observation
    s2 = [70] * len(obs)  # outer ring: Simulation

    # Create scatter plot
    im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.8, vmin=0, vmax=16, zorder=4)  # max  = 28
    im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.8, vmin=0, vmax=16, zorder=3)

    # Calculate the global mean of simulated and observed data
    global_mean_sim = round(np.nanmean(sim), 1)
    global_mean_obs = round(np.nanmean(obs), 1)
    global_std_sim = round(np.nanstd(sim), 1)
    global_std_obs = round(np.nanstd(obs), 1)

    # Display statistics as text annotations on the plot
    month_str = calendar.month_name[mon]
    ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.1f} ± {global_std_sim:.1f} µg/m$^3$',
            fontsize=12, fontname='Arial', transform=ax.transAxes)
    ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.1f} ± {global_std_obs:.1f} µg/m$^3$',
            fontsize=12, fontname='Arial', transform=ax.transAxes)
    ax.text(0.02, 0.05, f'{month_str}, 2018', fontsize=12, fontname='Arial', transform=ax.transAxes)

    # Plot title and colorbar
    plt.title(f'{species} Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
              fontsize=14, fontname='Arial')
    colorbar = plt.colorbar(im, label=f'{species} concentration (µg/m$^3$)',
                            orientation="vertical", pad=0.05, fraction=0.02)
    num_ticks = 5  # Adjust this value as needed
    colorbar.locator = plt.MaxNLocator(num_ticks)
    colorbar.update_ticks()
    font_properties = font_manager.FontProperties(family='Arial', size=12)
    colorbar.set_label(f'{species} concentration (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
    colorbar.ax.tick_params(axis='y', labelsize=10)
    plt.savefig(out_dir + '{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, mon), dpi=600)
    plt.show()