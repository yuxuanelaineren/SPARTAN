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

cres = 'C360'
year = 2018
species = 'SO4'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower())
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/'
################################################################################################
# Map SPARTAN and GCHP data
################################################################################################

for mon in range(1, 2):
    # Plot map using simulation data
    sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))

    plt.style.use('default')
    plt.figure(figsize=(10, 5))
    left = 0.1  # Adjust the left position
    bottom = 0.1  # Adjust the bottom position
    width = 0.8  # Adjust the width
    height = 0.8  # Adjust the height
    ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.BORDERS)

    ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())  # World map without Arctic and Antarctic region

    # Define the colormap
    cmap = WhGrYlRd
    cmap_reversed = cmap

    for face in range(6):
        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        # v = sim_df.BC.isel(nf=face)  # Change species as needed
        # v = sim_df.{}.isel(nf=face).format(species)
        # v = sim_df[species] .isel(nf=face)
        v = sim_df['BC'].isel(nf=face) / sim_df['PM25'].isel(nf=face)

        im = ax.pcolormesh(x, y, v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=100)

    # Read comparison data
    # compr_df = pd.read_csv(out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.csv'.format(cres, species, year, mon))
    compr_BC_df = pd.read_csv(out_dir + '{}_LUO_Sim_vs_SPARTAN_BC_{}{:02d}_MonMean.csv'.format(cres, year, mon))
    compr_PM25_df = pd.read_csv(out_dir + '{}_LUO_Sim_vs_SPARTAN_PM25_{}{:02d}_MonMean.csv'.format(cres, year, mon))

    compr_BC_notna = compr_BC_df[compr_BC_df.notna().all(axis=1)]
    compr_PM25_notna = compr_PM25_df [compr_PM25_df .notna().all(axis=1)]
    # compr_mon = compr_notna.loc[compr.month == mon]

    lon = compr_BC_notna.lon
    lat = compr_BC_notna.lat
    obs = compr_BC_notna.obs / compr_PM25_notna.obs
    sim = compr_BC_notna.sim

    # Define marker sizes
    s1 = [20] * len(obs)  # inner circle: Observation
    s2 = [70] * len(obs)  # Outer ring: Simulation
    # Create scatter plot
    im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=0.5, zorder=2) # max  = 15
    im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=0.5, zorder=1)

    # Calculate the global mean of simulated and observed data
    global_mean_sim = round(np.nanmean(sim), 1)
    global_mean_obs = round(np.nanmean(obs), 1)

    month_str = calendar.month_name[mon]
    ax.text(0.4, 0.12, 'Sim = {:.1f}'.format(global_mean_sim) + ' µg/m$^3$', fontsize=12, fontname='Arial',
            transform=ax.transAxes)
    ax.text(0.4, 0.05, 'Obs = {:.1f}'.format(global_mean_obs) + ' µg/m$^3$', fontsize=12, fontname='Arial',
            transform=ax.transAxes)
    ax.text(0.05, 0.05, '{}, 2018'.format(month_str), fontsize=12, fontname='Arial', transform=ax.transAxes)

    plt.title('BC/PM25 Comparison: GCHP-v13.4.1 {} vs SPARTAN'.format(cres.lower()), fontsize=14, fontname='Arial')
    plt.colorbar(im, label="BC/PM25", orientation="vertical",
                 pad=0.05, fraction=0.02)
    # plt.savefig(OutDir + '{}_Sim vs_SPARTAN_{}_{:02d}_MonMean.png'.format(cres, species, mon), dpi=500)
    plt.show()