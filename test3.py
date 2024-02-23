#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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

# Function to find matching rows and add 'Country' and 'City'
def find_and_add_location(lat, lon, site_df):
    for _, row in site_df.iterrows():
        if abs(row['Latitude'] - lat) <= 0.3 and abs(row['Longitude'] - lon) <= 0.3:
            return row['Country'], row['City']
    return None, None

# Main data processing loop
monthly_data = []
for month in range(1, 13):
    sim_df = xr.open_dataset(os.path.join(sim_dir, '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, month)), engine='netcdf4')
    obs_df = pd.read_excel(os.path.join(out_dir, 'BC_HIPS_SPARTAN.xlsx'), sheet_name='All')
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])

    obs_df = obs_df[obs_df['start_month'] == month]
    sim_lon = np.array(sim_df.lon).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lat).astype('float32')

    obs_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    obs_df.dropna(subset=[species], inplace=True)

    obs_lon = obs_df['Longitude']
    obs_lon[obs_lon > 180] -= 360
    obs_lat = obs_df['Latitude']

    # Calculate distance and match observations to simulation grid points
    coords_obs = np.vstack([obs_lat, obs_lon]).T
    coords_sim = np.vstack([sim_lat.ravel(), sim_lon.ravel()]).T
    distances = cdist(coords_obs, coords_sim, metric='euclidean')
    nearest_indices = distances.argmin(axis=1)
    nearest_sim_coords = coords_sim[nearest_indices]

    # Compile matched data
    matched_data = {
        'lat': nearest_sim_coords[:, 0],
        'lon': nearest_sim_coords[:, 1],
        'sim': sim_df[species].values.flatten()[nearest_indices],
        'obs': obs_df[species].values,
        'month': month
    }
    compr_df = pd.DataFrame(matched_data)
    compr_df[['country', 'city']] = compr_df.apply(lambda row: find_and_add_location(row['lat'], row['lon'], site_df), axis=1, result_type='expand')

    # Save monthly data
    outfile_path = os.path.join(out_dir, f'{cres}_{inventory}_{deposition}_Sim_vs_SPARTAN_{species}_{year}{month:02d}_MonMean.csv')
    compr_df.to_csv(outfile_path, index=False)

    monthly_data.append(compr_df)
    print(f'Month {month}: Data processed and saved.')

# Combine and analyze annual data
annual_df = pd.concat(monthly_data, ignore_index=True)
annual_summary = annual_df.groupby(['country', 'city']).agg({'sim': 'mean', 'obs': 'mean', 'lat': 'mean', 'lon': 'mean', 'month': 'count'}).reset_index().rename(columns={'month': 'num_obs'})

# Save annual summary
summary_file_path = os.path.join(out_dir, f'{cres}_{inventory}_{deposition}_Sim_vs_SPARTAN_{species}_{year}_Summary.xlsx')
with pd.ExcelWriter(summary_file_path, engine='openpyxl') as writer:
    annual_df.to_excel(writer, sheet_name='Monthly', index=False)
    annual_summary.to_excel(writer, sheet_name='Annual', index=False)

