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
# Other measurements 1: Combine measurement and GCHP dataset based on lat/lon
################################################################################################

# Function to find matching rows and add 'Country' and 'City'
def find_and_add_location(lat, lon):
    for index, row in obs_df.iterrows():
        if abs(row['Latitude'] - lat) <= 0.3 and abs(row['Longitude'] - lon) <= 0.3:
            return row['Country'], row['City']
    return None, None

# Create empty lists to store data for each month
monthly_data = []

# Loop through each month
for mon in range(1, 13):
    # Load simulation and observation data
    # sim_df = xr.open_dataset(sim_dir + 'WUCR3.LUO_WETDEP.C360.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(year, mon), engine='netcdf4') # EDGAR
    # sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') # HTAP
    # sim_df = xr.open_dataset('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/C720.LUO.PM25.RH35.NOx.O3.fromMonHourly.201801.MonMean.nc4', engine='netcdf4')
    sim_df = xr.open_dataset(
        sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon),
        engine='netcdf4')  # CEDS
    obs_df = pd.read_excel(out_dir + 'BC_CAWNET.xlsx')

    # Extract nf, Ydim, Xdim, lon/lat, buffer, and BC from simulation data
    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')

    sim_conc = np.array(sim_df[species]).reshape([6, 360, 360])
    buffer = 10

    # Drop NaN and infinite values from obs_conc
    obs_df = obs_df.replace([np.inf, -np.inf], np.nan)  # Convert infinite values to NaN
    obs_df = obs_df.dropna(subset=[species], thresh=1)

    # Extract lon/lat, BC, BC/PM25, and BC/SO4 from observation data
    obs_lon = obs_df['Longitude']
    obs_df.loc[obs_df['Longitude'] > 180, 'Longitude'] -= 360
    obs_lat = obs_df['Latitude']
    obs_conc = obs_df[species]

    # Find the nearest simulation lat/lon neighbors for each observation
    match_obs_lon = np.zeros(len(obs_lon))
    match_obs_lat = np.zeros(len(obs_lon))
    match_obs = np.zeros(len(obs_lon))
    match_sim_lon = np.zeros(len(obs_lon))
    match_sim_lat = np.zeros(len(obs_lon))
    match_sim = np.zeros(len(obs_lon))

    # Calculate distance between the observation and all simulation points using cdist
    for k in range(len(obs_lon)):
        # Spherical law of cosines:
        R = 6371  # Earth radius 6371 km
        latk = obs_lat.iloc[k]  # Use .iloc to access value by integer location
        lonk = obs_lon.iloc[k]  # Use .iloc to access value by integer location
        # Select simulation points within a buffer around the observation's lat/lon
        ind = np.where((sim_lon > lonk - buffer) & (sim_lon < lonk + buffer)
                       & (sim_lat > latk - buffer) & (sim_lat < latk + buffer))
        # Extract relevant simulation data
        sim_lonk = sim_lon[ind]
        sim_latk = sim_lat[ind]
        sim_conck = sim_conc[ind]
        # Calculate distance between the observation and selected simulation points
        dd = np.arccos(np.sin(latk * np.pi / 180) * np.sin(sim_latk * np.pi / 180) + \
                       np.cos(latk * np.pi / 180) * np.cos(sim_latk * np.pi / 180) * np.cos(
            (sim_lonk - lonk) * np.pi / 180)) * R
        ddmin = np.nanmin(dd)
        ii = np.where(dd == ddmin)
        # Use iloc to access the element by integer position
        match_obs[k] = obs_conc.iloc[k]
        match_sim[k] = np.nanmean(sim_conck[ii])
        match_sim_lat[k] = np.nanmean(sim_latk[ii])
        match_sim_lon[k] = np.nanmean(sim_lonk[ii])

    # Get unique lat/lon and observation data at the same simulation box
    coords = np.concatenate((match_sim_lat[:, None], match_sim_lon[:, None]), axis=1)
    coords_u, ind, ct = np.unique(coords, return_index=True, return_counts=True, axis=0)
    match_lon_u = match_sim_lon[ind]
    match_lat_u = match_sim_lat[ind]
    match_sim_u = match_sim[ind]
    # Calculate the monthly average observation data for each unique simulation box
    match_obs_u = np.zeros(len(ct))
    for i in range(len(ct)):
        irow = np.where((coords == coords_u[i]).all(axis=1))
        match_obs_u[i] = np.nanmean(match_obs[irow])

    # Drop rows with NaN values from the final data
    nanindex = np.argwhere(
        (np.isnan(match_lon_u) | np.isnan(match_lat_u) | np.isnan(match_sim_u) | np.isnan(match_obs_u))).squeeze()
    match_lon_u = np.delete(match_lon_u, nanindex)
    match_lat_u = np.delete(match_lat_u, nanindex)
    match_sim_u = np.delete(match_sim_u, nanindex)
    match_obs_u = np.delete(match_obs_u, nanindex)

    # Create DataFrame for current month
    columns = ['lat', 'lon', 'sim', 'obs', 'num_obs']
    compr_data = np.concatenate(
        (match_lat_u[:, None], match_lon_u[:, None], match_sim_u[:, None], match_obs_u[:, None], ct[:, None]), axis=1)
    compr_df = pd.DataFrame(data=compr_data, index=None, columns=columns)
    # Add a 'month' column to the DataFrame
    compr_df['month'] = mon

    # Apply the function to 'compr_df' and create new columns
    compr_df[['country', 'city']] = compr_df.apply(lambda row: find_and_add_location(row['lat'], row['lon']), axis=1,
                                                   result_type='expand')

    # Display the updated 'compr_df'
    print(compr_df)

    # Append data to the monthly_data list
    monthly_data.append(compr_df)

    # Calculate mean, sd, and max for simulated and observed concentrations
    mean_sim = np.nanmean(match_sim_u)
    sd_sim = np.nanstd(match_sim_u)
    max_sim = np.nanmax(match_sim_u)
    mean_obs = np.nanmean(match_obs_u)
    sd_obs = np.nanstd(match_obs_u)
    max_obs = np.nanmax(match_obs_u)
    # Print the results
    print(f'Simulated_{species}_in_{mon} Mean: {mean_sim:.2f}, SD: {sd_sim:.2f}, Max: {max_sim:.2f}')
    print(f'Observed_{species}_in_{mon} Mean: {mean_obs:.2f}, SD: {sd_obs:.2f}, Max: {max_obs:.2f}')

# Combine monthly data to create the annual DataFrame
annual_df = pd.concat(monthly_data, ignore_index=True)
# Add a 'month' column to the annual DataFrame
annual_df['month'] = annual_df['month'].astype(int)
# Calculate annual average for each site
annual_average_df = annual_df.groupby(['country', 'city']).agg({
    'sim': 'mean',
    'obs': 'mean',
    'num_obs': 'sum',
    'lat': 'mean',
    'lon': 'mean' }).reset_index()
with pd.ExcelWriter(out_dir + 'Other_Obs_vs {}_{}_{}_Sim_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year), engine='openpyxl') as writer:
    annual_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_average_df.to_excel(writer, sheet_name='Annual', index=False)
