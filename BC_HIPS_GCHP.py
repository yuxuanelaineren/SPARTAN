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

                # Convert the relevant columns to numeric to handle any non-numeric values
                HIPS_df['Mass_type'] = pd.to_numeric(HIPS_df['Mass_type'], errors='coerce')
                HIPS_df['BC_HIPS_ug'] = pd.to_numeric(HIPS_df['BC_HIPS_ug'], errors='coerce')
                HIPS_df['mass_ug'] = pd.to_numeric(HIPS_df['mass_ug'], errors='coerce')
                HIPS_df['Volume_m3'] = pd.to_numeric(HIPS_df['Volume_m3'], errors='coerce')
                HIPS_df['IC_SO4_ug'] = pd.to_numeric(HIPS_df['IC_SO4_ug'], errors='coerce')

                # Select PM2.5, rows where Mass_type is 1
                HIPS_df = HIPS_df.loc[HIPS_df['Mass_type'] == 1]

                # Convert 'start_year' to numeric and then to integers
                HIPS_df['start_year'] = pd.to_numeric(HIPS_df['start_year'], errors='coerce')
                HIPS_df['start_year'] = HIPS_df['start_year'].astype('Int64')  # 'Int64' allows for NaN values
                # Drop rows with NaN values in the 'start_year' column
                HIPS_df = HIPS_df.dropna(subset=['start_year'])

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

# Drop rows where 'BC_HIPS_ug' is NaN
HIPS_df = HIPS_df.dropna(subset=['BC_HIPS_ug'])

# Calculate BC concentrations, fractions, and BC/Sulfate
HIPS_df['BC_conc'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['Volume_m3']
HIPS_df['BC_frac'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['mass_ug']
HIPS_df['BC_Sulfate_ratio'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['IC_SO4_ug']

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
# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

for mon in range(1, 13):
    # Load simulation data
    sim_data = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))
    nf = np.array(sim_data.nf)
    Ydim = np.array(sim_data.Ydim)
    Xdim = np.array(sim_data.Xdim)
    sim_lon = np.array(sim_data.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_data.lats).astype('float32')
    sim_conc = np.array(sim_data['PM25'])
    buffer = 10

    # Load the observation data
    obs_lon = obs_df['Longitude']
    obs_df.loc[obs_df['Longitude'] > 180, 'Longitude'] -= 360
    obs_lat = obs_df['Latitude']
    obs_PM25 = obs_df['mass_ug']

    # Find the nearest simulation lat/lon neighbor at observation lat/lon
    match_obslon = np.zeros(len(obs_lon))
    match_obslat = np.zeros(len(obs_lon))
    match_obs = np.zeros(len(obs_lon))

    match_simlon = np.zeros(len(obs_lon))
    match_simlat = np.zeros(len(obs_lon))
    match_sim = np.zeros(len(obs_lon))

    for k in range(len(obs_lon)):
        # Spherical law of cosines:
        R = 6371  # Earth radius 6371 km
        latk = obs_lat[k]
        lonk = obs_lon[k]

        ind = np.where((sim_lon > lonk - buffer) & (sim_lon < lonk + buffer) & (sim_lat > latk - buffer) & (
                    sim_lat < latk + buffer))
        sim_lonk = sim_lon[ind]
        sim_latk = sim_lat[ind]
        sim_conck = sim_conc[ind]

        dd = np.arccos(np.sin(latk * np.pi / 180) * np.sin(sim_latk * np.pi / 180) + \
                       np.cos(latk * np.pi / 180) * np.cos(sim_latk * np.pi / 180) * np.cos(
            (sim_lonk - lonk) * np.pi / 180)) * R
        ddmin = np.nanmin(dd)
        ii = np.where(dd == ddmin)

        match_obs[k] = obs_PM25[k]
        match_sim[k] = np.nanmean(sim_conck[ii])
        match_simlat[k] = np.nanmean(sim_latk[ii])
        match_simlon[k] = np.nanmean(sim_lonk[ii])

    # Get unique lat/lon and average observation data at the same simulation box
    coords = np.concatenate((match_simlat[:, None], match_simlon[:, None]), axis=1)
    coords_u, ind, ct = np.unique(coords, return_index=True, return_counts=True, axis=0)
    match_lon_u = match_simlon[ind]
    match_lat_u = match_simlat[ind]
    match_sim_u = match_sim[ind]
    match_obs_u = np.zeros(len(ct))

    for i in range(len(ct)):
        irow = np.where((coords == coords_u[i]).all(axis=1))
        match_obs_u[i] = np.nanmean(match_obs[irow])

    # Drop NaN values in the final data
    nanindex = np.argwhere(
        (np.isnan(match_lon_u) | np.isnan(match_lat_u) | np.isnan(match_sim_u) | np.isnan(match_obs_u))).squeeze()
    match_lon_u = np.delete(match_lon_u, nanindex)
    match_lat_u = np.delete(match_lat_u, nanindex)
    match_sim_u = np.delete(match_sim_u, nanindex)
    match_obs_u = np.delete(match_obs_u, nanindex)

    # Create DataFrame and save to CSV
    columns = ['lat', 'lon', 'sim', 'obs', 'num_obs']
    data = np.concatenate(
        (match_lat_u[:, None], match_lon_u[:, None], match_sim_u[:, None], match_obs_u[:, None], ct[:, None]), axis=1)
    df = pd.DataFrame(data=data, index=None, columns=columns)
    outfile = out_dir + '{}_LUO_Sim_vs_CompileGM_noFILL_Obs_PM25_{}{:02d}_MonMean.csv'.format(cres, year, mon)
    df.to_csv(outfile)

################################################################################################
# Map SPARTAN and GCHP data
################################################################################################

for mon in range(1, 2):
    df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))

    plt.style.use('default')
    plt.figure(figsize=(5, 3))
    left = 0.1  # Adjust the left position as needed
    bottom = 0.2  # Adjust the bottom position as needed
    width = 0.8  # Adjust the width as needed
    height = 0.8  # Adjust the height as needed
    ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
    ax.coastlines()
    ax.set_global()
    ax.add_feature(cfeature.BORDERS)

    ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())  # World map without Arctic and Antarctic region

    cmap = WhGrYlRd
    cmap_reversed = cmap

    for face in range(6):
        x = df.corner_lons.isel(nf=face)
        y = df.corner_lats.isel(nf=face)
        v = df.PM25.isel(nf=face)  # Change species as needed
        im = ax.pcolormesh(x, y, v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=100)

    compr = pd.read_csv(out_dir + '{}_Sim_{}_vs_SPARTAN_{}_MonMean.csv'.format(cres, sim_year, species))
    compr_notna = compr[compr.notna().all(axis=1)]
    compr_mon = compr_notna.loc[compr.month == mon]
    lon = compr_mon.lon
    lat = compr_mon.lat
    obs = compr_mon.obs
    sim = compr_mon.sim
    s1 = [20] * len(obs)  # inner circle: Observation
    s2 = [70] * len(obs)  # Outer ring: Simulation
    im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=100, zorder=2)
    im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=100, zorder=1)

    global_mean_sim = round(np.nanmean(sim), 1)
    global_mean_obs = round(np.nanmean(obs), 1)

    month_str = calendar.month_name[mon]
    ax.text(0.6, 0.15, 'Sim = {:.1f}'.format(global_mean_sim) + ' \u03bcg/m$\mathregular{^3}$', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.6, 0.08, 'Obs = {:.1f}'.format(global_mean_obs) + ' \u03bcg/m$\mathregular{^3}$', fontsize=10,
            transform=ax.transAxes)
    ax.text(0.45, 0.1, '{}'.format(month_str), fontsize=10, transform=ax.transAxes)

    plt.title('GCHP-v13.4.1 {} vs SPARTAN in {}'.format(cres.lower(), month_str), fontsize=10)
    plt.colorbar(im, label="$PM_{2.5}$ concentrations (\u03bcg/m$\mathregular{^3}$)", orientation="horizontal",
                 pad=0.01, fraction=0.040)
    # plt.savefig(OutDir + '{}_Sim vs_SPARTAN_{}_{:02d}_MonMean.png'.format(cres, species, mon), dpi=500)
    plt.show()

