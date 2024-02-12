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

cres = 'C360'
year = 2015
species = 'BC'
inventory = 'EDGAR'
deposition = 'LUO'

# Set the directory path
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower()) # CEDS, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}-noLUO/monthly/'.format(cres.lower(), deposition) # HTAP, LUO
# sim_dir = '/Volumes/rvmartin/Active/dandan.z/AnalData/WUCR3-C360/' # EDGAR, LUO
sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/WUCR3-C360/' # EDGAR, LUO
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)

################################################################################################
# Extract BC_HIPS from master file and lon/lat from site.details
################################################################################################
# Create an empty list to store individual HIPS DataFrames
HIPS_dfs = []

# Iterate over each file in the directory
for filename in os.listdir(obs_dir):
    if filename.endswith('.csv'):
        # Read the data from the master file
        master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')

        # Specify the required columns
        HIPS_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                        'BC_HIPS_ug', 'IC_SO4_ug', 'IC_NO3_ug']
        # Check if all required columns are present
        if all(col in master_data.columns for col in HIPS_columns):
            # Remove leading/trailing whitespaces from column names
            master_data.columns = master_data.columns.str.strip() #Important!
            # Select the specified columns
            HIPS_df = master_data[HIPS_columns].copy()

            # Select PM2.5, rows where Mass_type is 1
            HIPS_df['Mass_type'] = pd.to_numeric(HIPS_df['Mass_type'], errors='coerce')
            HIPS_df = HIPS_df.loc[HIPS_df['Mass_type'] == 1]

            # Convert the relevant columns to numeric
            HIPS_df['BC_HIPS_ug'] = pd.to_numeric(HIPS_df['BC_HIPS_ug'], errors='coerce')
            HIPS_df['mass_ug'] = pd.to_numeric(HIPS_df['mass_ug'], errors='coerce')
            HIPS_df['Volume_m3'] = pd.to_numeric(HIPS_df['Volume_m3'], errors='coerce')
            HIPS_df['IC_SO4_ug'] = pd.to_numeric(HIPS_df['IC_SO4_ug'], errors='coerce')

            # Convert 'start_year' to numeric and then to integers
            # HIPS_df['start_year'] = pd.to_numeric(HIPS_df['start_year'], errors='coerce')
            # HIPS_df['start_year'] = HIPS_df['start_year'].astype('Int64')  # 'Int64' allows for NaN values

            # Drop rows with NaN values
            HIPS_df = HIPS_df.dropna(subset=['start_year'])
            HIPS_df = HIPS_df.dropna(subset=['Volume_m3'])
            HIPS_df = HIPS_df.dropna(subset=['BC_HIPS_ug']) # Don't forget to change
            HIPS_df = HIPS_df.dropna(subset=['IC_SO4_ug']) # Don't forget to change

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

# Assuming your DataFrame is named obs_df
site_counts = HIPS_df.groupby('Site')['FilterID'].count()

# Print the number of rows for each site
for site, count in site_counts.items():
    print(f"{site}: {count} rows")

# Calculate BC concentrations, fractions, and BC/Sulfate
HIPS_df['BC'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['Volume_m3']
HIPS_df['PM25'] = HIPS_df['mass_ug'] / HIPS_df['Volume_m3']
HIPS_df['SO4'] = HIPS_df['IC_SO4_ug'] / HIPS_df['Volume_m3']
HIPS_df['BC_PM25'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['mass_ug']
HIPS_df['BC_SO4'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['IC_SO4_ug']
HIPS_df['BC_PM25_NO3'] = HIPS_df['BC_HIPS_ug'] / (HIPS_df['mass_ug'] - HIPS_df['IC_NO3_ug'])

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

# Function to find matching rows and add 'Country' and 'City'
def find_and_add_location(lat, lon):
    for index, row in site_df.iterrows():
        if abs(row['Latitude'] - lat) <= 0.3 and abs(row['Longitude'] - lon) <= 0.3:
            return row['Country'], row['City']
    return None, None

# Create empty lists to store data for each month
monthly_data = []

# Loop through each month
for mon in range(1, 13):
    # Load simulation and observation data
    sim_df = xr.open_dataset(sim_dir + 'WUCR3.LUO_WETDEP.C360.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(year, mon), engine='netcdf4') # EDGAR
    # sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') # HTAP
    # sim_df = xr.open_dataset('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/C720.LUO.PM25.RH35.NOx.O3.fromMonHourly.201801.MonMean.nc4', engine='netcdf4')

    obs_df = pd.read_excel(out_dir + 'HIPS_SPARTAN.xlsx')
    # Filter obs_df based on 'start_month'
    obs_df = obs_df[obs_df['start_month'] == mon]
    # Display information about the dataset
    # print(sim_df)
    # Display information about data variables
    # print("Data variables:", sim_df.data_vars)
    # sim_df.close()

    # Extract nf, Ydim, Xdim, lon/lat, buffer, and BC from simulation data
    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')

    sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']
    sim_df['BC_PM25_NO3'] = sim_df['BC'] / (sim_df['PM25'] - sim_df['NIT'])
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

    # Get unique lat/lon and average observation data at the same simulation box
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

    # Save monthly CSV file
    outfile = out_dir + '{}_{}_{}_Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.csv'.format(cres, inventory, deposition, species, year, mon)
    compr_df.to_csv(outfile, index=False)  # Set index=False to avoid writing row indices to the CSV file

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
with pd.ExcelWriter(out_dir + '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year), engine='openpyxl') as writer:
    annual_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_average_df.to_excel(writer, sheet_name='Annual', index=False)

sim_df.close()

################################################################################################
# Map SPARTAN and GCHP data
################################################################################################
# Map SPARTAN and GCHP data
for mon in range(1, 13):
    # Plot map using simulation data
    sim_df = xr.open_dataset(sim_dir + 'WUCR3.LUO_WETDEP.C360.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(year, mon),
                             engine='netcdf4')  # EDGAR
    sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']
    sim_df['BC_PM25_NO3'] = sim_df['BC'] / (sim_df['PM25'] - sim_df['NIT'])

    plt.style.use('default')
    plt.figure(figsize=(10, 5))
    left = 0.05  # Adjust the left position
    bottom = 0.02  # Adjust the bottom position
    width = 0.85  # Adjust the width
    height = 0.95  # Adjust the height
    ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
    ax.coastlines(color=(0.4, 0.4, 0.4))  # Set the color of coastlines to dark grey
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))  # Set the color of borders to dark grey
    ax.set_global()

    ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

    # Define the colormap
    cmap = WhGrYlRd
    cmap_reversed = cmap

    vmax = 8  # 8 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4, 0.3 for BC_PM25_NO3

    # Plot data for each face
    for face in range(6):
        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        im = ax.pcolormesh(x, y, v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

    # Read comparison data
    compar_filename = f'{cres}_{inventory}_{deposition}_Sim_vs_SPARTAN_{species}_{year}{mon:02d}_MonMean.csv'
    compar_df = pd.read_csv(os.path.join(out_dir, compar_filename))
    compar_notna = compar_df[compar_df.notna().all(axis=1)]
    lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim

    # Define marker sizes
    s1 = [40] * len(obs)  # inner circle: Observation
    s2 = [120] * len(obs)  # outer ring: Simulation

    # Create scatter plot
    im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.8, vmin=0, vmax=vmax, zorder=4)
    im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                    linewidth=0.8, vmin=0, vmax=vmax, zorder=3)

    # Calculate the global mean of simulated and observed data
    global_mean_sim = np.nanmean(sim)
    global_mean_obs = np.nanmean(obs)
    global_std_sim = np.nanstd(sim)
    global_std_obs = np.nanstd(obs)

    # Display statistics as text annotations on the plot
    month_str = calendar.month_name[mon]
    ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.2f}',
            fontsize=12, fontname='Arial', transform=ax.transAxes)
    ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.2f}',
            fontsize=12, fontname='Arial', transform=ax.transAxes)
    ax.text(0.02, 0.05, f'{month_str}, 2015', fontsize=12, fontname='Arial', transform=ax.transAxes)

    # Plot title and colorbar
    plt.title(f'{species} Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=14, fontname='Arial') # PM$_{{2.5}}$
    colorbar = plt.colorbar(im, orientation="vertical", pad=0.05, fraction=0.02)
    num_ticks = 5
    colorbar.locator = plt.MaxNLocator(num_ticks)
    colorbar.update_ticks()
    font_properties = font_manager.FontProperties(family='Arial', size=12)
    # colorbar.set_label(f'BC/Sulfate', labelpad=10, fontproperties=font_properties)
    colorbar.set_label(f'{species} concentration (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
    colorbar.ax.tick_params(axis='y', labelsize=10)
    # plt.savefig(out_dir + '{}_{}_{}Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.tiff'.format(cres, inventory, deposition, species, year, mon), dpi=600)
    plt.show()

################################################################################################
# Map SPARTAN and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(10, 5))
left = 0.05  # Adjust the left position
bottom = 0.02  # Adjust the bottom position
width = 0.85  # Adjust the width
height = 0.95  # Adjust the height
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

# Define the colormap
cmap = WhGrYlRd
cmap_reversed = cmap
vmax = 8  # 8 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

# Accumulate data for each face over the year
annual_v = None

for face in range(6):
    for mon in range(1, 13):
        sim_df = xr.open_dataset(
            sim_dir + 'WUCR3.LUO_WETDEP.C360.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(year, mon),
            engine='netcdf4')  # EDGAR
        sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
        sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']
        sim_df['BC_PM25_NO3'] = sim_df['BC'] / (sim_df['PM25'] - sim_df['NIT'])

        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        if annual_v is None:
            annual_v = v
        else:
            annual_v += v

    # Calculate the annual average
    annual_v /= 12

    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim

# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Observation
s2 = [120] * len(obs)  # outer ring: Simulation

# Create scatter plot
im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                linewidth=0.8, vmin=0, vmax=vmax, zorder=4)
im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                linewidth=0.8, vmin=0, vmax=vmax, zorder=3)

# Calculate the global mean of simulated and observed data
global_mean_sim = np.nanmean(sim)
global_mean_obs = np.nanmean(obs)
global_std_sim = np.nanstd(sim)
global_std_obs = np.nanstd(obs)

# Display statistics as text annotations on the plot
month_str = calendar.month_name[mon]
ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.2f}',
        fontsize=12, fontname='Arial', transform=ax.transAxes)
ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.2f}',
        fontsize=12, fontname='Arial', transform=ax.transAxes)
ax.text(0.02, 0.05, f'2015', fontsize=12, fontname='Arial', transform=ax.transAxes)

# Plot title and colorbar
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN',
            fontsize=14, fontname='Arial') # PM$_{{2.5}}$
colorbar = plt.colorbar(im, orientation="vertical", pad=0.05, fraction=0.02)
num_ticks = 5
colorbar.locator = plt.MaxNLocator(num_ticks)
colorbar.update_ticks()
font_properties = font_manager.FontProperties(family='Arial', size=12)
# colorbar.set_label(f'BC/Sulfate', labelpad=10, fontproperties=font_properties)
colorbar.set_label(f'{species} concentration (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
colorbar.ax.tick_params(axis='y', labelsize=10)
# plt.savefig(out_dir + '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_AnnualMean.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()

################################################################################################
# Create scatter plot for monthly and annual data
################################################################################################
# Read the file
annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, species, year)), sheet_name='Mon')
# annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, species, year)), sheet_name='Annual')

# Drop rows where BC is greater than 1
annual_df = annual_df.loc[annual_df['obs'] <= 20]

# Print the names of each city
unique_cities = annual_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
region_mapping = {
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke', 'Mexico City', 'Fajardo',
                      'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'Australia': ['Melbourne']
}
# Define custom palette for each region with 5 shades for each color
# Define custom palette for each region with 5 shades for each color
region_colors = {
    'Central Asia': [
        (1, 0.42, 0.70), (0.8, 0.52, 0.7), (1, 0.4, 0.4), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ],  # Pink shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.2, 0.2), (1, 0.48, 0.41), (1, 0.6, 0.6)
    ],  # Red shades
    'East Asia': [
        (1, 0.64, 0), (1, 0.55, 0.14), (1, 0.63, 0.48), (1, 0.74, 0.61), (1, 0.85, 0.73), (1, 0.96, 0.85)
    ],  # Orange shades
    'North America': [
        (0, 0, 0.5), (0, 0, 0.8), (0, 0, 1), (0.39, 0.58, 0.93), (0.54, 0.72, 0.97), (0.68, 0.85, 0.9)
    ],  # Blue shades
    'South America': [
        (0.58, 0.1, 0.81), (0.9, 0.4, 1), (0.66, 0.33, 0.83), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    ],  # Purple shades
    'Africa': [
        (0, 0.5, 0), (0, 0.8, 0), (0, 1, 0), (0.56, 0.93, 0.56), (0.56, 0.93, 0.56), (0.8, 0.9, 0.8)
    ],  # Green shades
    'Australia': [
        (0.6, 0.4, 0.2)
    ]  # Brown
}

def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)  # Default to black if city is not found
# Create an empty list to store the city_palette for each city
city_palette = []
city_color_match = []
# Iterate over each unique city and map it to a gradient
for city in unique_cities:
    city_color = map_city_to_color(city)
    if city_color is not None:
        city_palette.append(city_color)
        city_color_match.append({'city': city, 'color': city_color})  # Store both city name and color
print("City Palette:", city_palette)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=annual_df, hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Create a function to determine the index of a city in region_mapping
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')  # If city is not found, place it at the end
def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None
# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)
# Create legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))
legend_labels = [city['city'] for city in sorted_city_color_match]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=city['color'], markersize=8, label=city['city']) for city in sorted_city_color_match]
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)
# legend.get_frame().set_linewidth(0.0)

# legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)

# Set title, xlim, ylim, ticks, labels
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
          fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 12])
plt.ylim([-0.5, 12])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([0, 4, 8, 12], fontname='Arial', size=18)
plt.yticks([0, 4, 8, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = annual_df['obs']
y = annual_df['obs']
plt.plot([annual_df['obs'].min(), annual_df['obs'].max()], [annual_df['obs'].min(), annual_df['obs'].max()],
         color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(annual_df)
plt.text(0.60, 0.4, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18)
# plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=22)
plt.text(0.85, 0.05, f'2018', transform=scatterplot.transAxes, fontsize=18)
# plt.text(0.05, 0.81, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)

# Perform linear regression with NaN handling
mask = ~np.isnan(annual_df['obs']) & ~np.isnan(annual_df['sim'])
slope, intercept, r_value, p_value, std_err = stats.linregress(annual_df['obs'][mask], annual_df['sim'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='obs', y='sim', data=annual_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.60, 0.45, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=18)

plt.xlabel('SPARTAN Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('GCHP Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
# plt.savefig('/Users/renyuxuan/Downloads/' + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)

plt.show()
################################################################################################
# Other measurements 1: Combine measurement and GCHP dataset based on lat/lon
################################################################################################
# Create empty lists to store data for each month
monthly_data = []

# Loop through each month
for mon in range(1, 13):
    # Load simulation and observation data
    sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4')
    # sim_df = xr.open_dataset('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/C720.LUO.PM25.RH35.NOx.O3.fromMonHourly.201801.MonMean.nc4', engine='netcdf4')
    obs_df = pd.read_excel(out_dir + 'BC_CAWNET.xlsx')
    # Filter obs_df based on 'start_month'
    # obs_df = obs_df[obs_df['start_month'] == mon]
    # Display information about the dataset
    # print(sim_df)
    # Display information about data variables
    # print("Data variables:", sim_df.data_vars)
    # sim_df.close()

    # Extract nf, Ydim, Xdim, lon/lat, buffer, and BC from simulation data
    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')

    sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']
    sim_conc = np.array(sim_df[species])
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

    # Display the updated 'compr_df'
    print(compr_df)

    # Save monthly CSV file
    outfile = out_dir + '{}_LUO_Sim_vs_CAWNET_{}_{}{:02d}_MonMean.csv'.format(cres, species, year, mon)
    compr_df.to_csv(outfile, index=False)  # Set index=False to avoid writing row indices to the CSV file

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
annual_average_df = annual_df.groupby(['lat', 'lon']).agg({'sim': 'mean', 'obs': 'mean', 'num_obs': 'sum'}).reset_index()
with pd.ExcelWriter(out_dir + '{}_LUO_Sim_vs_CAWNET_{}_{}_Summary.xlsx'.format(cres, species, year), engine='openpyxl') as writer:
    annual_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_average_df.to_excel(writer, sheet_name='Annual', index=False)
# Save annual CSV file
# annual_file = out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}_AnnualMean.csv'.format(cres, species, year)
# annual_df.to_csv(annual_file, index=False)  # Set index=False to avoid writing row indices to the CSV file
sim_df.close()

################################################################################################
# # Other measurements 2: Map measurement and GCHP data for the entire year
################################################################################################
# Map measurement and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(10, 5))
left, bottom, width, height = 0.05, 0.02, 0.85, 0.9
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())

# Set the extent to cover China
ax.set_extent([65, 140, 15, 55], crs=ccrs.PlateCarree()) # westernmost, easternmost, southernmost, northernmost
# Add map features
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
# ax.set_global()
# ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

# Define the colormap
cmap = WhGrYlRd
cmap_reversed = cmap
vmax = 8  # 8 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

# Accumulate data for each face over the year
annual_v = None

for face in range(6):
    for mon in range(1, 13):
        sim_df = xr.open_dataset(f'{sim_dir}{cres}.LUO.PM25.RH35.NOx.O3.{year}{mon:02d}.MonMean.nc4')
        sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
        sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']

        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        if annual_v is None:
            annual_v = v
        else:
            annual_v += v

    # Calculate the annual average
    annual_v /= 12

    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_CAWNET_{}_{}_Summary.xlsx'.format(cres, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim

# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Observation
s2 = [120] * len(obs)  # outer ring: Simulation

# Create scatter plot
im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                linewidth=0.8, vmin=0, vmax=vmax, zorder=4, marker='s') # marker='^', triangles
im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=cmap_reversed, edgecolor='black',
                linewidth=0.8, vmin=0, vmax=vmax, zorder=3, marker='s')

# Additional SPARTAN Beijing data
point_lon = 116.3340073
point_lat = 39.98311996
point_sim = 8.927127918
point_obs = 2.454504375

# Plot the additional SPARTAN point as a circle
ax.scatter(x=point_lon, y=point_lat, c=point_sim, s=120, transform=ccrs.PlateCarree(),
           cmap=cmap_reversed, edgecolor='black', linewidth=0.8, vmin=0, vmax=vmax, zorder=5, marker='o')
ax.scatter(x=point_lon, y=point_lat, c=point_obs, s=40, transform=ccrs.PlateCarree(),
           cmap=cmap_reversed, edgecolor='black', linewidth=0.8, vmin=0, vmax=vmax, zorder=5, marker='o')

# Calculate the global mean of simulated and observed data
global_mean_sim = np.nanmean(sim)
global_mean_obs = np.nanmean(obs)
global_std_sim = np.nanstd(sim)
global_std_obs = np.nanstd(obs)

# Display statistics as text annotations on the plot
month_str = calendar.month_name[mon]
ax.text(0.8, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.2f}',
        fontsize=12, fontname='Arial', transform=ax.transAxes)
ax.text(0.8, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.2f}',
        fontsize=12, fontname='Arial', transform=ax.transAxes)
ax.text(0.02, 0.05, f'2018', fontsize=12, fontname='Arial', transform=ax.transAxes)

# Plot title and colorbar
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} v.s. CAWNET',
            fontsize=14, fontname='Arial') # PM$_{{2.5}}$
# plt.title(f'{species} Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN', fontsize=14, fontname='Arial')
colorbar = plt.colorbar(im, orientation="vertical", pad=0.05, fraction=0.02)
num_ticks = 5
colorbar.locator = plt.MaxNLocator(num_ticks)
colorbar.update_ticks()
font_properties = font_manager.FontProperties(family='Arial', size=12)
# colorbar.set_label(f'BC/Sulfate', labelpad=10, fontproperties=font_properties)
colorbar.set_label(f'{species} concentration (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
colorbar.ax.tick_params(axis='y', labelsize=10)
# plt.savefig(out_dir + '{}_Sim_vs_CAWNET_{}_{}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
plt.show()



################################################################################################
# Create scatter plot for difference in sim and obs vs elevation
################################################################################################
# Set the directory path
sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/'
# Read the file
# diff_df = pd.read_excel(os.path.join(sim_dir, 'c360_BC/C360_LUO_Sim_vs_SPARTAN_BC_2018_Summary.xlsx'), sheet_name = 'Annual')
diff_df = pd.read_excel(os.path.join(out_dir, 'c360_c720_BC_2018_01.xlsx'))

# Print the names of each city
unique_cities = diff_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
region_mapping = {
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke',
                      'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena', 'Fajardo', 'Mexico City'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'Australia': ['Melbourne']
}
# Define custom palette for each region with 5 shades for each color
region_colors = {
    'Central Asia': [
        (1, 0.42, 0.70), (0.8, 0.52, 0.7), (1, 0.4, 0.4), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ],  # Pink shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.2, 0.2), (1, 0.48, 0.41), (1, 0.6, 0.6)
    ],  # Red shades
    'East Asia': [
        (1, 0.64, 0), (1, 0.55, 0.14), (1, 0.63, 0.48), (1, 0.74, 0.61), (1, 0.85, 0.73), (1, 0.96, 0.85)
    ],  # Orange shades
    'North America': [
        (0, 0, 0.5), (0, 0, 0.8), (0, 0, 1), (0.39, 0.58, 0.93), (0.54, 0.72, 0.97), (0.68, 0.85, 0.9)
    ],  # Blue shades
    'South America': [
        (0.58, 0.1, 0.81), (0.9, 0.4, 1), (0.66, 0.33, 0.83), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    ],  # Purple shades
    'Africa': [
        (0, 0.5, 0), (0, 0.8, 0), (0, 1, 0), (0.56, 0.93, 0.56), (0.56, 0.93, 0.56), (0.8, 0.9, 0.8)
    ],  # Green shades
    'Australia': [
        (0.6, 0.4, 0.2)
    ]  # Brown
}

def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)  # Default to black if city is not found
# Create an empty list to store the city_palette for each city
city_palette = []
city_color_match = []
# Iterate over each unique city and map it to a gradient
for city in unique_cities:
    city_color = map_city_to_color(city)
    if city_color is not None:
        city_palette.append(city_color)
        city_color_match.append({'city': city, 'color': city_color})  # Store both city name and color
print("City Palette:", city_palette)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
# Plot 'difference_c360' as circles
sns.scatterplot(x='Elevation_meters', y='difference_c360', data=diff_df, hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='o')
# Plot 'difference_c720' as triangles
sns.scatterplot(x='Elevation_meters', y='difference_c720', data=diff_df, hue='city', palette=city_palette, s=80, alpha=1, ax=ax, edgecolor='k', marker='^')

ax.set_facecolor('white')
border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid


# Create a function to determine the index of a city in region_mapping
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')  # If city is not found, place it at the end
def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None
# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)
# Create city legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))
legend_labels = [city['city'] for city in sorted_city_color_match]
legend_handles_city = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=city['color'], markersize=8, label=city['city']) for city in sorted_city_color_match]
legend_city = ax.legend(handles=legend_handles_city, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)

# Set x-axis scale to logarithmic
plt.xscale('log')
# Set title, xlim, ylim, ticks, labels
plt.title(f'BC Comparison: Difference vs Elevation',
          fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([2, 4000])
plt.ylim([-10, 15])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([10, 100, 1000, 3000], fontname='Arial', size=18)
plt.yticks([-10, -5, 0, 5, 10, 15], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add y = 0 with grey dash
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(diff_df)
plt.text(0.08, 0.9, f'N = {num_points}', transform=ax.transAxes, fontsize=18)
# plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=22)
plt.text(0.65, 0.05, f'January, 2018', transform=ax.transAxes, fontsize=18)
# plt.text(0.05, 0.81, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)

plt.xlabel('Elevation (m)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Difference (Observation - Simulation) (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
plt.savefig(out_dir + 'Scatter_c720_c360_BC_Difference_Elevation_2018.tiff', dpi=600)
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
# plt.savefig('/Users/renyuxuan/Downloads/' + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)

plt.show()

# seasonal variations
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
CEDS_sim = [15.93606377, 13.06744576, 7.127143383, 6.255723953, 4.879863262, 3.123033524, 3.557436228, 3.227097988, 4.803154945, 12.49798489, 17.76980782, 20.32411766]
obs = [2.735662251, 3.27318941, 2.73875873, 2.362111952, 2.140101034, 1.838316092, 1.5547228, 1.885276454, 2.371057107, 2.851534884, 4.014037007, 1.673341111]
HTAP_sim = [10.87263489, 11.16574287, 12.18006897, 6.738657475, 6.194628239, 5.789536953, 5.534196854, 6.513328552, 7.129979134, 7.787933826, 14.91756535, 12.3012619]

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot
plt.scatter(months, CEDS_sim, label='CEDS Simulation', color='red', s=80, alpha=1, edgecolor='k', marker='o')
plt.scatter(months, HTAP_sim, label='HTAP Observation', color='blue', s=80, alpha=1, edgecolor='k', marker='o')
plt.scatter(months, obs, label='Observation', color='green', s=80, alpha=1, edgecolor='k', marker='^')

border_width = 1
plt.ylim([0, 22])
plt.xticks([0, 2, 4, 6, 8, 10, 12], fontname='Arial', size=18)
plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)

# Add labels and title
plt.title('Seasonal variations in BC Concentration in Beijing', fontsize=16, fontname='Arial', y=1.03)
plt.xlabel('Month', fontsize=18, color='black', fontname='Arial')
plt.ylabel('BC concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Customize legend
legend = plt.legend(fontsize=16)
legend.get_frame().set_edgecolor('black')  # Set legend border color
legend.get_frame().set_linewidth(border_width)  # Set legend border width

# Set legend font to Arial
prop = fm.FontProperties(family='Arial', size=16)
plt.setp(legend.get_texts(), fontproperties=prop)

# plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/c360_noLUO_BC/Scatter_c360_CEDS_HTAP_BC_MonMean.tiff', dpi=600)

# Show the plot
plt.show()
