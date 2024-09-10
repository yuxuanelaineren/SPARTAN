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
import geopandas as gpd

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
# Map CEDS emissions in Beijing site
################################################################################################
# Path to the NetCDF file
CEDS_dir = '/Volumes/rvmartin/Active/GEOS-Chem-shared/ExtData/HEMCO/CEDS/v2024-06/2019/CEDS_BC_0.1x0.1_2019.nc'
CEDS_df = xr.open_dataset(CEDS_dir)
# print(CEDS_df)

# Extract emissions data and average over time
emission_data = {key: CEDS_df[key].mean(dim='time') for key in
                 ['BC_agr', 'BC_ene', 'BC_ind', 'BC_rco', 'BC_shp', 'BC_slv', 'BC_tra', 'BC_wst']}
lat = CEDS_df['lat']
lon = CEDS_df['lon']

def plot_emissions(data, title, key, vmin=0, vmax=5e-11):
    """Plot emissions data on a Cartopy map with a circle for Beijing site."""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines(color=(0.4, 0.4, 0.4))
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.set_extent([115, 118, 39.3, 41.3], crs=ccrs.PlateCarree())  # Focus on Beijing

    # Define the colormap
    colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.7, 0, 0)]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)

    im = ax.pcolormesh(lon, lat, data, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    # Add colorbar
    cbar_axes = inset_axes(ax, width='50%', height='3%', bbox_to_anchor=(-0.25, -1.05, 1, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
    cbar = plt.colorbar(im, cax=cbar_axes, orientation='horizontal')
    font_properties = font_manager.FontProperties(family='Arial', size=12)
    cbar.set_ticks([0, 1e-11, 2e-11, 3e-11, 4e-11, 5e-11], fontproperties=font_properties)
    cbar.ax.set_title('')
    cbar.ax.title.set_visible(False)
    cbar.ax.set_xlabel('')
    cbar.ax.set_xlabel('BC Emissions (kg/m$^2$/s)', labelpad=2, fontproperties=font_properties)
    cbar.ax.xaxis.set_label_position('bottom')  # Ensure label is at the bottom
    cbar.ax.tick_params(axis='x', labelsize=12)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1)

    # Add circle for SPARTAN Beijing site
    site_lat, site_lon = 40.004, 116.326
    circle = patches.Circle((site_lon, site_lat), radius=0.015, edgecolor='black', facecolor='none',
                            transform=ccrs.PlateCarree())
    # Add Beijing city border
    beijing_shapefile = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/Beijing_border_2020/Beijing-2020.shp'
    gdf = gpd.read_file(beijing_shapefile)
    gdf = gdf.to_crs(crs=ccrs.PlateCarree().proj4_init)  # Convert CRS to match map
    ax.add_geometries(gdf.geometry, ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=1)

    ax.add_patch(circle)
    plt.suptitle(f'{title}', fontsize=14, fontproperties=font_manager.FontProperties(family='Arial'), y=0.95)
    plt.subplots_adjust(top=0.95)  # Adjust the top margin to move the plot up

    filename = f'FigSX_Beijing_CEDS0.1v2024-06_{key}.tiff'
    plt.savefig(out_dir + filename, dpi=600, bbox_inches='tight')
    plt.show()

def extract_emission_at_site(data, lat, lon, site_lat, site_lon):
    """Extract BC emissions data at a specific site location."""
    # Convert xarray DataArrays to NumPy arrays
    lat_np = lat.values.flatten()  # Ensure it's a 1D array
    lon_np = lon.values.flatten()  # Ensure it's a 1D array

    # Find the closest latitude and longitude index
    lat_idx = np.argmin(np.abs(lat_np - site_lat))
    lon_idx = np.argmin(np.abs(lon_np - site_lon))

    # Check the dimensions
    print(f'Latitude Index: {lat_idx}, Longitude Index: {lon_idx}')

    # Print the emissions data at the site
    for key in data:
        # Extract the emission value at the site
        emission_value = data[key].isel(lat=lat_idx, lon=lon_idx).values
        print(f'{key} emissions at ({site_lat}, {site_lon}): {emission_value} kg/m$^2$/s')


extract_emission_at_site(emission_data, lat, lon, 40.004, 116.326)

# # Define titles for each emission category
# emission_titles = {
#     'BC_ind': 'CEDS 0.1 BC Industrial Emissions',
#     'BC_rco': 'CEDS 0.1 BC Residential, Commercial, Other Combustion Emissions',
#     'BC_tra': 'CEDS 0.1 BC Transportation Emissions',
#     'BC_shp': 'CEDS 0.1 BC International Shipping Emissions',
#     'BC_slv': 'CEDS 0.1 BC Solvents production and application',
#     'BC_wst': 'CEDS 0.1 BC Waste Emissions',
#     'BC_agr': 'CEDS 0.1 BC Agriculture Emissions',
#     'BC_ene': 'CEDS 0.1 BC Energy Emissions'
# }
#
# # Plot each category with the appropriate title
# for key, title in emission_titles.items():
#     if key in emission_data:
#         plot_emissions(emission_data[key], title)
#
# Plot each category
plot_emissions(emission_data['BC_ind'], 'CEDS0.1 BC Industrial Emissions', 'BC_ind')
plot_emissions(emission_data['BC_rco'], 'CEDS0.1 BC Residential, Commercial, Other Combustion Emissions', 'BC_rco')
plot_emissions(emission_data['BC_tra'], 'CEDS0.1 BC Transportation Emissions', 'BC_tra')



