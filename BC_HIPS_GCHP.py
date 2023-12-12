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

cres = 'C360'
year = 2018
species = 'BC'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower())
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/'
# out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/output-{}/'.format(cres.lower())

################################################################################################
# Extract BC_HIPS from master file and lon/lat from site.details
################################################################################################

# # Extract SPARTAN data from the master file folder
# Create an empty list to store individual HIPS DataFrames
HIPS_dfs = []

# Iterate over each file in the directory
for filename in os.listdir(obs_dir):
    if filename.endswith('.csv'):
        # Read the data from the master file
        master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')

        # Specify the required columns
        HIPS_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                        'BC_HIPS_ug', 'IC_SO4_ug']

        # Check if all required columns are present
        if all(col in master_data.columns for col in HIPS_columns):

            # Select the specified columns
            HIPS_df = master_data[HIPS_columns].copy()

            # Select PM2.5, rows where Mass_type is 1
            HIPS_df = HIPS_df.loc[HIPS_df['Mass_type'] == 1]

            # Convert the relevant columns to numeric to handle any non-numeric values
            HIPS_df['Mass_type'] = pd.to_numeric(HIPS_df['Mass_type'], errors='coerce')
            HIPS_df['BC_HIPS_ug'] = pd.to_numeric(HIPS_df['BC_HIPS_ug'], errors='coerce')
            HIPS_df['mass_ug'] = pd.to_numeric(HIPS_df['mass_ug'], errors='coerce')
            HIPS_df['Volume_m3'] = pd.to_numeric(HIPS_df['Volume_m3'], errors='coerce')
            HIPS_df['IC_SO4_ug'] = pd.to_numeric(HIPS_df['IC_SO4_ug'], errors='coerce')

            # Convert 'start_year' to numeric and then to integers
            HIPS_df['start_year'] = pd.to_numeric(HIPS_df['start_year'], errors='coerce')
            HIPS_df['start_year'] = HIPS_df['start_year'].astype('Int64')  # 'Int64' allows for NaN values

            # Drop rows with NaN values
            HIPS_df = HIPS_df.dropna(subset=['start_year'])
            HIPS_df = HIPS_df.dropna(subset=['BC_HIPS_ug'])
            HIPS_df = HIPS_df.dropna(subset=['Volume_m3'])

            # Extract the site name from the filename
            site_name = filename.split('_')[0]
            # Add the site name as a column in the selected data
            HIPS_df["Site"] = [site_name] * len(HIPS_df)

            # Append the current HIPS_df to the list
            HIPS_dfs.append(HIPS_df)
        else:
            print(f"Skipping {filename} because not all required columns are present.")

# Concatenate all HIPS DataFrames into a single DataFrame
HIPS_df = pd.concat(HIPS_dfs, ignore_index=True)

# Calculate BC concentrations, fractions, and BC/Sulfate
HIPS_df['BC_conc'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['Volume_m3']
HIPS_df['BC_PM25_ratio'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['mass_ug']
HIPS_df['BC_SO4_ratio'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['IC_SO4_ug']

# Read Site name and lon/lat from Site_detail.xlsx
site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'),
                        usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])

# Merge the dataframes based on the "Site" and "Site_Code" columns
obs_df = pd.merge(HIPS_df, site_df, how="left", left_on="Site", right_on="Site_Code")

# Drop the duplicate "Site_Code" column
obs_df.drop("Site_Code", axis=1, inplace=True)

# Write to excel file
with pd.ExcelWriter(os.path.join(out_dir, "HIPS_SPARTAN.xlsx"), engine='openpyxl') as writer:
    # Write the HIPS data to the 'HIPS_All' sheet
    obs_df.to_excel(writer, sheet_name='HIPS_All', index=False)

################################################################################################
# Combine SPARTAN and GCHP dataset based on lat/lon
################################################################################################
# Loop through each month
for mon in range(1, 13):
    # Load simulation data
    sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))

    # Display information about the dataset
    # print(sim_df)
    # Access variables within the dataset
    # variable_sim = sim_df['variable_name']
    # sim_df.close()

    # Extract nf, Ydim, Xdim, lon/lat, buffer, and BC from simulation data
    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')
    sim_conc = np.array(sim_df['BC'])  # Don't forget to change
    buffer = 10

    # Drop NaN and infinite values from obs_conc
    obs_df = obs_df.replace([np.inf, -np.inf], np.nan)  # Convert infinite values to NaN
    obs_df = obs_df.dropna(subset=['BC_conc'])

    # Extract lon/lat, BC, BC/PM, and BC/SO4 from observation data
    obs_lon = obs_df['Longitude']
    obs_df.loc[obs_df['Longitude'] > 180, 'Longitude'] -= 360
    obs_lat = obs_df['Latitude']
    obs_conc = obs_df['BC_conc']  # Don't forget to change
    # obs_BC_PM25 = obs_df['BC_PM25_ratio']
    # obs_BC_SO4 = obs_df['BC_SO4_ratio']

    # Find the nearest simulation lat/lon neighbors for each observation
    match_obs_lon = np.zeros(len(obs_lon))
    match_obs_lat = np.zeros(len(obs_lon))
    match_obs = np.zeros(len(obs_lon))

    match_sim_lon = np.zeros(len(obs_lon))
    match_sim_lat = np.zeros(len(obs_lon))
    match_sim = np.zeros(len(obs_lon))

    # Calculate distance between the observation and all simulation points
    for k in range(len(obs_lon)):
        # Spherical law of cosines:
        R = 6371  # Earth radius 6371 km
        latk = obs_lat.iloc[k]  # Use .iloc to access value by integer location
        lonk = obs_lon.iloc[k]  # Use .iloc to access value by integer location
        # Select simulation points within a buffer around the observation's lat/lon
        ind = np.where((sim_lon > lonk - buffer) & (sim_lon < lonk + buffer)
                       & (sim_lat > latk - buffer) & (sim_lat < latk + buffer))
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

    # Get unique lat/lon and average observation data at the same simulation box
    coords = np.concatenate((match_sim_lat[:, None], match_sim_lon[:, None]), axis=1)
    coords_u, ind, ct = np.unique(coords, return_index=True, return_counts=True, axis=0)
    match_lon_u = match_sim_lon[ind]
    match_lat_u = match_sim_lat[ind]
    match_sim_u = match_sim[ind]
    match_obs_u = np.zeros(len(ct))
    # Calculate the average observation data for each unique simulation box
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

    # Create DataFrame and save to CSV
    columns = ['lat', 'lon', 'sim', 'obs', 'num_obs']
    compr_data = np.concatenate(
        (match_lat_u[:, None], match_lon_u[:, None], match_sim_u[:, None], match_obs_u[:, None], ct[:, None]), axis=1)
    compr_df = pd.DataFrame(data=compr_data, index=None, columns=columns)
    outfile = out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.csv'.format(cres, species, year, mon)
    compr_df.to_csv(outfile)

    # Extract lat/lon and country, city from obs_df
    obs_lat_lon_country = obs_df[['Latitude', 'Longitude', 'Country', 'City']]

    # Get lat/lon from compr_data
    compr_lat_lon = compr_data[:, :2]

    # Match lat/lon with obs_df
    matched_indices = []
    for lat_lon in compr_lat_lon:
        lat, lon = lat_lon
        index = ((obs_lat_lon_country[['Latitude', 'Longitude']] == lat_lon).all(axis=1)).idxmax()
        matched_indices.append(index)

    # Assign country information to compr_data
    compr_data_with_country = np.concatenate(
        [compr_data, obs_lat_lon_country.loc[matched_indices, 'Country'].values[:, None]], axis=1)

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

################################################################################################
# Map SPARTAN and GCHP data
################################################################################################

for mon in range(1, 2):
    # Plot map using simulation data
    sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))

    plt.style.use('default')
    plt.figure(figsize=(10, 6))
    left = 0.1  # Adjust the left position as needed
    bottom = 0.2  # Adjust the bottom position as needed
    width = 0.8  # Adjust the width as needed
    height = 0.8  # Adjust the height as needed
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
        v = sim_df[species].isel(nf=face)

        im = ax.pcolormesh(x, y, v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=100)

    # Read comparison data
    compr_df = pd.read_csv(out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.csv'.format(cres, species, year, mon))
    # compr = pd.read_csv(out_dir + '{}_LUO_Sim_{}_vs_SPARTAN_{}_MonMean.csv'.format(cres, year, species))
    compr_notna = compr_df[compr_df.notna().all(axis=1)]
    # compr_mon = compr_notna.loc[compr.month == mon]

    lon = compr_notna.lon
    lat = compr_notna.lat
    obs = compr_notna.obs
    sim = compr_notna.sim

    # Define marker sizes
    s1 = [20] * len(obs)  # inner circle: Observation
    s2 = [70] * len(obs)  # Outer ring: Simulation
    # Create scatter plot
    im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=15, zorder=2)
    im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=15, zorder=1)

    # Calculate the global mean of simulated and observed data
    global_mean_sim = round(np.nanmean(sim), 1)
    global_mean_obs = round(np.nanmean(obs), 1)

    month_str = calendar.month_name[mon]
    ax.text(0.6, 0.15, 'Sim = {:.1f}'.format(global_mean_sim) + ' µg/m$^3$', fontsize=12, fontname='Arial',
            transform=ax.transAxes)
    ax.text(0.6, 0.08, 'Obs = {:.1f}'.format(global_mean_obs) + ' µg/m$^3$', fontsize=12, fontname='Arial',
            transform=ax.transAxes)
    ax.text(0.15, 0.1, '{}'.format(month_str), fontsize=12, fontname='Arial', transform=ax.transAxes)

    plt.title('Comparison of simulated (GCHP-v13.4.1 {}) and observed BC {}, 2018'.format(cres.lower(), month_str), fontsize=14)
    plt.colorbar(im, label="Black Carbon concentrations (µg/m$^3$)", orientation="vertical",
                 pad=0.05, fraction=0.08)
    # plt.savefig(OutDir + '{}_Sim vs_SPARTAN_{}_{:02d}_MonMean.png'.format(cres, species, mon), dpi=500)
    plt.show()

