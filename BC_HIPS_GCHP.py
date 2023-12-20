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
year = 2018
species = 'BC_SO4'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower())
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}/'.format(cres.lower(), species)

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
                        'BC_HIPS_ug', 'IC_SO4_ug']
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
        if abs(row['Latitude'] - lat) <= 1 and abs(row['Longitude'] - lon) <= 1:
            return row['Country'], row['City']
    return None, None

# Create empty lists to store data for each month
monthly_data = []

# Loop through each month
for mon in range(1, 13):
    # Load simulation and observation data
    sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4')
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
    outfile = out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.csv'.format(cres, species, year, mon)
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

# Calculate annual average for 'sim' and 'obs', and sum for 'num_obs' for each site
annual_average_df = annual_df.groupby(['Country', 'City']).agg({'sim': 'mean', 'obs': 'mean', 'num_obs': 'sum'}).reset_index()

with pd.ExcelWriter(out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}_AnnualMean.xlsx'.format(cres, species, year), engine='openpyxl') as writer:
    annual_df.to_excel(writer, sheet_name='All', index=False)
    annual_average_df.to_excel(writer, sheet_name='Average', index=False)

# Save annual CSV file
# annual_file = out_dir + '{}_LUO_Sim_vs_SPARTAN_{}_{}_AnnualMean.csv'.format(cres, species, year)
# annual_df.to_csv(annual_file, index=False)  # Set index=False to avoid writing row indices to the CSV file
sim_df.close()

################################################################################################
# Map SPARTAN and GCHP data
################################################################################################
# Map SPARTAN and GCHP data
for mon in range(1, 13):
    # Plot map using simulation data
    sim_df = xr.open_dataset(f'{sim_dir}{cres}.LUO.PM25.RH35.NOx.O3.{year}{mon:02d}.MonMean.nc4')
    sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']

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

    vmax = 2  # 10 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

    # Plot data for each face
    for face in range(6):
        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        im = ax.pcolormesh(x, y, v, cmap=cmap_reversed, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

    # Read comparison data
    compar_filename = f'{cres}_LUO_Sim_vs_SPARTAN_{species}_{year}{mon:02d}_MonMean.csv'
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
    ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.3f}',
            fontsize=12, fontname='Arial', transform=ax.transAxes)
    ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.3f}',
            fontsize=12, fontname='Arial', transform=ax.transAxes)
    ax.text(0.02, 0.05, f'{month_str}, 2018', fontsize=12, fontname='Arial', transform=ax.transAxes)

    # Plot title and colorbar
    plt.title(f'BC/Sulfate Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
              fontsize=14, fontname='Arial') # PM$_{{2.5}}$
    # plt.title(f'{species}$ Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN', fontsize=14, fontname='Arial')
    colorbar = plt.colorbar(im, orientation="vertical", pad=0.05, fraction=0.02)
    num_ticks = 5
    colorbar.locator = plt.MaxNLocator(num_ticks)
    colorbar.update_ticks()
    font_properties = font_manager.FontProperties(family='Arial', size=12)
    colorbar.set_label(f'BC/Sulfate', labelpad=10, fontproperties=font_properties)
    # colorbar.set_label(f'{species} concentration (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
    colorbar.ax.tick_params(axis='y', labelsize=10)
    # plt.savefig(out_dir + '{}_Sim_vs_SPARTAN_{}_{}_{:02d}_MonMean.tiff'.format(cres, species, year, mon), dpi=600)
    plt.show()

################################################################################################
# Map SPARTAN and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(10, 5))
left = 0.1  # Adjust the left position
bottom = 0.1  # Adjust the bottom position
width = 0.8  # Adjust the width
height = 0.8  # Adjust the height
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

# Define the colormap
cmap = WhGrYlRd
cmap_reversed = cmap
vmax = 2  # 10 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

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
compar_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_AnnualMean.xlsx'.format(cres, species, year)),
                          sheet_name='Average')
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
ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.3f}',
        fontsize=12, fontname='Arial', transform=ax.transAxes)
ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.3f}',
        fontsize=12, fontname='Arial', transform=ax.transAxes)
ax.text(0.02, 0.05, f'{month_str}, 2018', fontsize=12, fontname='Arial', transform=ax.transAxes)

# Plot title and colorbar
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
            fontsize=14, fontname='Arial') # PM$_{{2.5}}$
# plt.title(f'{species}$ Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN', fontsize=14, fontname='Arial')
colorbar = plt.colorbar(im, orientation="vertical", pad=0.05, fraction=0.02)
num_ticks = 5
colorbar.locator = plt.MaxNLocator(num_ticks)
colorbar.update_ticks()
font_properties = font_manager.FontProperties(family='Arial', size=12)
# colorbar.set_label(f'BC/Sulfate', labelpad=10, fontproperties=font_properties)
colorbar.set_label(f'{species} concentration (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
colorbar.ax.tick_params(axis='y', labelsize=10)
# plt.savefig(out_dir + '{}_Sim_vs_SPARTAN_{}_{}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
plt.show()

################################################################################################
# Create scatter plot for monthly data
################################################################################################
# Read the file
annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_AnnualMean.xlsx'.format(cres, species, year)), sheet_name='All')

# Drop rows where BC is greater than 1
annual_df = annual_df.loc[annual_df['obs'] <= 20]

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=annual_df, hue='City', s=25, alpha=1, ax=ax)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Modify legend background color and position
legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)
# legend.get_frame().set_linewidth(0.0)  # remove legend border
legend.get_texts()[0].set_fontname("Arial")  # set fontname of the first label

# Set title, xlim, ylim, ticks, labels
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
          fontsize=14, fontname='Arial', y=1.03)  # PM$_{{2.5}}$

plt.xlim([-0.5, 16])
plt.ylim([-0.5, 16])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=14)
plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=14)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = annual_df['obs']
y = annual_df['obs']
plt.plot([annual_df['sim'].min(), annual_df['sim'].max()], [annual_df['sim'].min(), annual_df['sim'].max()],
         color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(annual_df)
plt.text(0.70, 0.3, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)
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
    plt.text(0.70, 0.35, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=14)

plt.xlabel('SPARTAN Black Carbon (µg/m$^3$)', fontsize=14, color='black', fontname='Arial')
plt.ylabel('GCHP Black Carbon (µg/m$^3$)', fontsize=14, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(out_dir + '{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)
plt.show()

