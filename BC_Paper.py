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
import matplotlib.colors as mcolors

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
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)

################################################################################################
# Extract BC_HIPS from masterfile and lon/lat from site.details
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(obs_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350',
        'MXMC-0006', 'NGIL-0309'
    ]
    HIPS_dfs = []
    for filename in os.listdir(obs_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(obs_dir, filename), encoding='ISO-8859-1')
            HIPS_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                            'BC_HIPS_ug', 'Flags']
            if all(col in master_data.columns for col in HIPS_columns):
                # Select the specified columns
                master_data.columns = master_data.columns.str.strip()
                HIPS_df = master_data[HIPS_columns].copy()
                # Exclude specific FilterID values
                HIPS_df = HIPS_df[~HIPS_df['FilterID'].isin(excluded_filters)]
                # Select PM2.5
                HIPS_df['Mass_type'] = pd.to_numeric(HIPS_df['Mass_type'], errors='coerce')
                HIPS_df = HIPS_df.loc[HIPS_df['Mass_type'] == 1]
                # Convert the relevant columns to numeric
                HIPS_df[['BC_HIPS_ug', 'mass_ug', 'Volume_m3', 'start_year']] = HIPS_df[
                    ['BC_HIPS_ug', 'mass_ug', 'Volume_m3', 'start_year']].apply(pd.to_numeric, errors='coerce')
                # Select year 2019 - 2023
                HIPS_df = HIPS_df[HIPS_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]
                # Drop rows with NaN values
                HIPS_df = HIPS_df.dropna(subset=['start_year', 'Volume_m3', 'BC_HIPS_ug'])
                HIPS_df = HIPS_df[HIPS_df['Volume_m3'] > 0]  # Exclude rows where Volume_m3 is 0
                HIPS_df = HIPS_df[HIPS_df['BC_HIPS_ug'] > 0]  # Exclude rows where HIPS_BC is 0
                # Calculate BC concentrations, fractions, and BC/Sulfate
                HIPS_df['BC'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['Volume_m3']
                HIPS_df['PM25'] = HIPS_df['mass_ug'] / HIPS_df['Volume_m3']
                HIPS_df['BC_PM25'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['mass_ug']
                # Extract the site name and add as a column
                site_name = filename.split('_')[0]
                HIPS_df["Site"] = [site_name] * len(HIPS_df)
                # Append the current HIPS_df to the list
                HIPS_dfs.append(HIPS_df)
            else:
                print(f"Skipping {filename} because not all required columns are present.")
    return pd.concat(HIPS_dfs, ignore_index=True)

# Main script
if __name__ == "__main__":
    # Read data
    HIPS_df = read_master_files(obs_dir)
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])
    obs_df = pd.merge(HIPS_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)

    # Write HIPS data to Excel
    with pd.ExcelWriter(os.path.join(out_dir, "BC_HIPS_SPARTAN.xlsx"), engine='openpyxl', mode='w') as writer:
        obs_df.to_excel(writer, sheet_name='All', index=False)

    # Writ summary statistics to Excel
    site_counts = obs_df.groupby('Site')['FilterID'].count()
    for site, count in site_counts.items():
        print(f"{site}: {count} rows")
    summary_df = obs_df.groupby(['Country', 'City'])['BC'].agg(['count', 'mean', 'median', 'std'])
    summary_df['stderr'] = summary_df['std'] / np.sqrt(summary_df['count']).pow(0.5)
    summary_df.rename(columns={'count': 'num_obs', 'mean': 'bc_mean', 'median': 'bc_median', 'std': 'bc_stdv', 'stderr': 'bc_stderr'},
        inplace=True)
    with pd.ExcelWriter(os.path.join(out_dir, "BC_HIPS_SPARTAN.xlsx"), engine='openpyxl', mode='a') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=True)

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
    # sim_df = xr.open_dataset(sim_dir + 'WUCR3.LUO_WETDEP.C360.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(year, mon), engine='netcdf4') # EDGAR
    # sim_df = xr.open_dataset(sim_dir + '{}.LUO.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') # HTAP
    # sim_df = xr.open_dataset(sim_dir + 'C720.LUO.PM25.RH35.NOx.O3.fromMonHourly.201801.MonMean.nc4', engine='netcdf4') # c720, HTAP
    sim_df = xr.open_dataset(sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon), engine='netcdf4') # CEDS
    obs_df = pd.read_excel(out_dir + 'BC_HIPS_SPARTAN.xlsx', sheet_name='All')
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'),
                            usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])

    # Filter obs_df based on 'start_month'
    obs_df = obs_df[obs_df['start_month'] == mon]

    # Extract nf, Ydim, Xdim, lon/lat, buffer, and BC from simulation data
    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')

    sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
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
    obs_year = obs_df['start_year']

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
    compr_df['year'] = obs_year

    # Apply the function to 'compr_df' and create new columns
    compr_df[['country', 'city']] = compr_df.apply(lambda row: find_and_add_location(row['lat'], row['lon']), axis=1,
                                                   result_type='expand')
    # Display the updated 'compr_df'
    print(compr_df)

    # Save monthly CSV file
    # outfile = os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}{:02d}_MonMean.csv'.format(cres, inventory, deposition, species, year, mon))
    # compr_df.to_csv(outfile, index=False)  # Set index=False to avoid writing row indices to the CSV file

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
# Map SPARTAN and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(12, 5))
left = 0.03  # Adjust the left position
bottom = 0.01  # Adjust the bottom position
width = 0.94  # Adjust the width
height = 0.9  # Adjust the height
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())

# Define the colormap
cmap = WhGrYlRd
custom_cmap = cmap # Blue to red

# Define colormap (from white to dark red through yellow and orange)
colors = ['#f7f7f7',   # light gray
          '#ffff00',   # yellow
          '#ffA500',   # orange
          '#ff4500',   # red-orange
          '#ff0000',   # red
          '#8b0000',   # dark red
          '#4d0000']   # even darker red

# Create a LinearSegmentedColormap
cmap_name = 'custom_heat'
n_bins = 100  # Increase for smoother transition
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

vmax = 8  # 8 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

# Accumulate data for each face over the year
annual_v = None

for face in range(6):
    for mon in range(1, 13):
        sim_df = xr.open_dataset(
            sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon),
            engine='netcdf4') # CEDS

        sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
        sim_df['BC_SO4'] = sim_df['BC'] / sim_df['SO4']
        sim_df['BC_PM25_NO3'] = sim_df['BC'] / (sim_df['PM25'] - sim_df['NIT'])

        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)

        if annual_v is None:
            annual_v = v
        else:
            annual_v = annual_v + v

    # Calculate the annual average
    annual_v /= 12
    annual_v = annual_v.squeeze()
    print(x.shape, y.shape, annual_v.shape)

    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_v, cmap=custom_cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim

# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Observation
s2 = [120] * len(obs)  # outer ring: Simulation

# Create scatter plot
im = ax.scatter(x=lon, y=lat, c=obs, s=s1, transform=ccrs.PlateCarree(), cmap=custom_cmap, edgecolor='black',
                linewidth=1, vmin=0, vmax=vmax, zorder=4)
im = ax.scatter(x=lon, y=lat, c=sim, s=s2, transform=ccrs.PlateCarree(), cmap=custom_cmap, edgecolor='black',
                linewidth=1, vmin=0, vmax=vmax, zorder=3)

# Calculate the global mean of simulated and observed data
global_mean_sim = np.nanmean(sim)
global_mean_obs = np.nanmean(obs)
global_std_sim = np.nanstd(sim)
global_std_obs = np.nanstd(obs)

# Display statistics as text annotations on the plot
month_str = calendar.month_name[mon]
ax.text(0.4, 0.12, f'Sim = {global_mean_sim:.2f} ± {global_std_sim:.2f}',
        fontsize=16, fontname='Arial', transform=ax.transAxes)
ax.text(0.4, 0.05, f'Obs = {global_mean_obs:.2f} ± {global_std_obs:.2f}',
        fontsize=16, fontname='Arial', transform=ax.transAxes)
ax.text(0.9, 0.05, f'2019', fontsize=16, fontname='Arial', transform=ax.transAxes)
plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN',
            fontsize=16, fontname='Arial') # PM$_{{2.5}}$

# Create an inset axes for the color bar at the left middle of the plot
colorbar_axes = inset_axes(ax,
                           width="2%",
                           height="60%",
                           bbox_to_anchor=(-0.95, -0.35, 1, 1),  # (x, y, width, height) relative to top-right corner
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
cbar = plt.colorbar(im, cax=colorbar_axes, orientation="vertical")
num_ticks = 5
cbar.locator = plt.MaxNLocator(num_ticks)
cbar.update_ticks()
font_properties = font_manager.FontProperties(family='Arial', size=14)
cbar.set_label(f'{species} (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
cbar.ax.tick_params(axis='y', labelsize=14)

# plt.savefig(out_dir + 'Fig2_{}_{}_{}_Sim_vs_SPARTAN_{}_{}_AnnualMean.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()

################################################################################################
# Create scatter plot for c360 vs c720, colored by region
################################################################################################
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
def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)
def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_marker = region_markers[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Marker: {assigned_marker}")
            return assigned_marker
    print(f"City not found in any region: {city}")
    return (0, 0, 0)
def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city)
            assigned_marker = region_markers[region][city_index % len(region_markers[region])]
            return assigned_marker
    return None
# Read the file
compr_df = pd.read_csv(os.path.join(out_dir + 'C720_HTAP_LUO_Sim_vs_C360_HTAP_LUO_Sim_BC_201801_MonMean.csv'))

# Drop rows where BC is greater than 1
# compr_df = compr_df.loc[compr_df['obs'] <= 20]

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
region_mapping = {
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke', 'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena', 'Fajardo', 'Mexico City'],
    'Australia': ['Melbourne'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
}
region_mapping = {region: [city for city in cities if city in unique_cities] for region, cities in region_mapping.items()}

# Define custom palette for each region with 5 shades for each color, https://rgbcolorpicker.com/0-1
region_colors = {
    'North America': [
        (0, 0, 0.6),  (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9)
    ],  # Blue shades
    'Central Asia': [
        (0.58, 0.1, 0.81), (0.66, 0.33, 0.83), (0.9, 0.4, 1), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    ],  # Purple shades
    'Australia': [
        (0.6, 0.4, 0.2)
    ],  # Brown
    'East Asia': [
        (0, 0.5, 0), (0, 0.8, 0), (0, 1, 0), (0.56, 0.93, 0.56), (0.8, 0.9, 0.8)
    ],  # Green shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.4, 0.4), (0.9, 0.6, 0.6)
    ],  # Red shades
    'Africa': [
        (1, 0.4, 0), (1, 0.6, 0.14), (1, 0.63, 0.48), (1, 0.85, 0.73), (1, 0.96, 0.85)
    ], # Orange shades
    'South America': [
        (1, 0.16, 0.827), (1, 0.42, 0.70), (0.8, 0.52, 0.7), (0.961, 0.643, 0.804), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ]  # Pink shades
}

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

# Define custom palette for each region with 5 shades for each color
# markers = ['o', 's', '^', 'v', '<', '>', 'D', '*', 'H', '+', 'x', 'P', 'p', 'X', '1', '2', '3', '4']
# ['o', 'H', 'p', 's', '^', 'P']
region_markers = {
    'North America': ['o', '^', 's', 'p', 'H', '*'],
    'Australia': ['o', '^', 's', 'p', 'H', '*'],
    'East Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Central Asia': ['o', '^', 's', 'p', 'H', '*'],
    'South Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Africa': ['o', '^', 's', 'p', 'H', '*'],
    'South America': ['o', '^', 's', 'p', 'H', '*'],
}

# Create an empty list to store the city_marker for each city
city_marker = []
city_marker_match = []

# Iterate over each unique city and map it to a marker
for city in unique_cities:
    marker = map_city_to_marker(city)
    if marker is not None:
        city_marker.append(marker)
        city_marker_match.append({'city': city, 'marker': marker})

print("City Marker:", city_marker)

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 2.4]
x_range_2 = [2.4, compr_df['obs'].max()]
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='c360', y='c720', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city',  markers=city_marker)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)

# Create legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))

# Create legend handles with both color and marker for each city
legend_handles = []
for city_info in sorted_city_color_match:
    city = city_info['city']
    color = city_info['color']
    marker = map_city_to_marker(city)
    if marker is not None:
        handle = plt.Line2D([0], [0], marker=marker, color=color, linestyle='', markersize=8, label=city)
        legend_handles.append(handle)

# Create legend with custom handles
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 12]) # 11 for edgar
plt.ylim([-0.5, 12])
plt.xticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
plt.yticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['c360']
y = compr_df['c360']
plt.plot([compr_df['c360'].min(), 11.5], [compr_df['c360'].min(), 11.5],
         color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['c360'][mask], compr_df['c720'][mask])
# Plot regression lines
sns.regplot(x='c360', y='c720', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
plt.text(0.65, 0.05, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('C360 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('C720 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS3_Scatter_c720_vs_c360_201801.svg', dpi=300)

plt.show()

################################################################################################
# Create scatter plot for annual data (color blue and red)
################################################################################################
# Read the file
# compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Mon')
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 2.4]
x_range_2 = [2.4, compr_df['obs'].max()]

# Define custom blue and red colors
blue_colors = [(0.7, 0.76, 0.9),  (0.431, 0.584, 1), (0.4, 0.5, 0.9), (0, 0.27, 0.8),  (0, 0, 1), (0, 0, 0.6)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]

# Create custom colormap
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', blue_colors)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', red_colors)

# Create a custom color palette mapping each city to a color based on observed values
def map_city_to_color(city, obs):
    if x_range_1[0] <= obs <= x_range_1[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()) - 1)
        return blue_cmap(obs_index)
    elif x_range_2[0] <= obs <= x_range_2[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()) - 1)
        return red_cmap(obs_index)
    else:
        return 'black'
city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]

# Sort the cities in the legend based on observed values
sorted_cities = sorted(compr_df['city'].unique(), key=lambda city: compr_df.loc[compr_df['city'] == city, 'obs'].iloc[0])

# Classify 'city' based on 'region'

def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None

region_mapping = {
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke', 'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena', 'Fajardo', 'Mexico City'],
    'Australia': ['Melbourne'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
}
region_mapping = {region: [city for city in cities if city in unique_cities] for region, cities in region_mapping.items()}

region_markers = {
    'North America': ['o', 'o', 'o', 'p', 'H', '*'],
    'Australia': ['o', '^', 's', 'p', 'H', '*'],
    'East Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Central Asia': ['o', '^', 's', 'p', 'H', '*'],
    'South Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Africa': ['o', 'o', 'o', 'o', 'o', 'o'],
    'South America': ['o', '^', 's', 'p', 'H', '*'],
}
# Create an empty list to store the city_marker for each city
city_marker = []
city_marker_match = []

def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            if region == 'North America':
                return 'd'
            elif region == 'Australia':
                return '*'
            elif region == 'East Asia':
                return '^'
            elif region == 'Central Asia':
                return 'p'
            elif region == 'South Asia':
                return 's'
            elif region == 'Africa':
                return 'o'
            elif region == 'South America':
                return 'o'
            else:
                return 'o'  # Default marker style
    print(f"City not found in any region: {city}")
    return 'o'

# Iterate over each unique city and map it to a marker
for city in unique_cities:
    marker = map_city_to_marker(city)
    if marker is not None:
        city_marker.append(marker)
        city_marker_match.append({'city': city, 'marker': marker})

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# Create scatter plot
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city',  markers=city_marker)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1.5
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 13.5]) # 14 for edgar
plt.ylim([-0.5, 13.5])
plt.xticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
plt.yticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['obs']
y = compr_df['obs']
plt.plot([compr_df['obs'].min(), compr_df['obs'].max()], [compr_df['obs'].min(), compr_df['obs'].max()],
         color='grey', linestyle='--', linewidth=1)

# Perform linear regression for the first segment
mask_1 = (compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])
slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(compr_df['obs'][mask_1], compr_df['sim'][mask_1])
# Perform linear regression for the second segment
mask_2 = (compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])
slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(compr_df['obs'][mask_2], compr_df['sim'][mask_2])
# Plot regression lines
sns.regplot(x='obs', y='sim', data=compr_df[mask_1],
            scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display_1 = abs(intercept_1)
intercept_display_2 = abs(intercept_2)
intercept_sign_1 = '-' if intercept_1 < 0 else '+'
intercept_sign_2 = '-' if intercept_2 < 0 else '+'
plt.text(0.05, 0.81, f'y = {slope_1:.2f}x {intercept_sign_1} {intercept_display_1:.2f}\n$r^2$ = {r_value_1 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='blue')
plt.text(0.05, 0.56, f'y = {slope_2:.2f}x {intercept_sign_2} {intercept_display_2:.2f}\n$r^2$ = {r_value_2 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='red')

# Add the number of data points for each segment
num_points_1 = mask_1.sum()
plt.text(0.05, 0.75, f'N = {num_points_1}', transform=scatterplot.transAxes, fontsize=18, color='blue')
num_points_2 = mask_2.sum()
plt.text(0.05, 0.50, f'N = {num_points_2}', transform=scatterplot.transAxes, fontsize=18, color='red')

# Set labels
plt.xlabel('Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig1_b_r_Scatter_{}_{}_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, inventory, deposition, species, year), dpi=600)

plt.show()

################################################################################################
# Create scatter plot for annual data (color blue and red) with one line
################################################################################################
# Read the file
# compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Mon')
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 2.4]
x_range_2 = [2.4, compr_df['obs'].max()]
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]

# Define custom blue and red colors
blue_colors = [(0.7, 0.76, 0.9),  (0.431, 0.584, 1), (0.4, 0.5, 0.9), (0, 0.27, 0.8),  (0, 0, 1), (0, 0, 0.6)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]

# Create custom colormap
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', blue_colors)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', red_colors)

# Create a custom color palette mapping each city to a color based on observed values
def map_city_to_color(city, obs):
    if x_range_1[0] <= obs <= x_range_1[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()) - 1)
        return blue_cmap(obs_index)
    elif x_range_2[0] <= obs <= x_range_2[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()) - 1)
        return red_cmap(obs_index)
    else:
        return 'black'
city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]

# Sort the cities in the legend based on observed values
sorted_cities = sorted(compr_df['city'].unique(), key=lambda city: compr_df.loc[compr_df['city'] == city, 'obs'].iloc[0])

# Classify 'city' based on 'region'
def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None

region_mapping = {
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke', 'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena', 'Fajardo', 'Mexico City'],
    'Australia': ['Melbourne'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
}
region_mapping = {region: [city for city in cities if city in unique_cities] for region, cities in region_mapping.items()}

region_markers = {
    'North America': ['o', 'o', 'o', 'p', 'H', '*'],
    'Australia': ['o', '^', 's', 'p', 'H', '*'],
    'East Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Central Asia': ['o', '^', 's', 'p', 'H', '*'],
    'South Asia': ['o', '^', 's', 'p', 'H', '*'],
    'Africa': ['o', 'o', 'o', 'o', 'o', 'o'],
    'South America': ['o', '^', 's', 'p', 'H', '*'],
}
# Create an empty list to store the city_marker for each city
city_marker = []
city_marker_match = []

def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            if region == 'North America':
                return 'd'
            elif region == 'Australia':
                return '*'
            elif region == 'East Asia':
                return '^'
            elif region == 'Central Asia':
                return 'p'
            elif region == 'South Asia':
                return 's'
            elif region == 'Africa':
                return 'o'
            elif region == 'South America':
                return 'o'
            else:
                return 'o'  # Default marker style
    print(f"City not found in any region: {city}")
    return 'o'

# Iterate over each unique city and map it to a marker
for city in unique_cities:
    marker = map_city_to_marker(city)
    if marker is not None:
        city_marker.append(marker)
        city_marker_match.append({'city': city, 'marker': marker})

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# Create scatter plot
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city',  markers=city_marker)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1.5
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 12]) # 11 for edgar
plt.ylim([-0.5, 12])
plt.xticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
plt.yticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['obs']
y = compr_df['obs']
plt.plot([compr_df['obs'].min(), 11.5], [compr_df['obs'].min(), 11.5],
         color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['obs'][mask], compr_df['sim'][mask])
# Plot regression lines
sns.regplot(x='obs', y='sim', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
plt.text(0.75, 0.05, f'N = {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
plt.savefig(out_dir + 'Fig_b_r_Scatter_{}_{}_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.svg'.format(cres, inventory, deposition, species, year), dpi=300)

plt.show()

################################################################################################
# Other: Summarize IMPROVE EC data
################################################################################################
# Read the data
other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
IMPROVE_df = pd.read_excel(other_obs_dir + 'IMPROVE_EC_2019_raw.xlsx', sheet_name='Data')

# Convert 'Date' column to datetime format
IMPROVE_df['Date'] = pd.to_datetime(IMPROVE_df['Date'], format='%m/%d/%Y')
IMPROVE_df['Year'] = IMPROVE_df['Date'].dt.year
IMPROVE_df['Month'] = IMPROVE_df['Date'].dt.month
IMPROVE_df['Day'] = IMPROVE_df['Date'].dt.day
# Filter data for the year 2019
IMPROVE_df = IMPROVE_df[IMPROVE_df['Year'] == 2019]
# Filter valid data, Flag = V0
IMPROVE_df = IMPROVE_df[IMPROVE_df['ECf_Status'] == 'V0']
# Filter POC = 1
IMPROVE_df = IMPROVE_df[IMPROVE_df['POC'] == 1]

# Group by 'SiteName', 'Latitude', 'Longitude', and calculate monthly average 'ECf_Val'
EC_mon_df = IMPROVE_df.groupby(['SiteName', 'Latitude', 'Longitude', 'Month'])['ECf_Val'].agg(['count', 'mean']).reset_index()
EC_mon_df.columns = ['SiteName', 'Latitude', 'Longitude', 'Month', 'num_obs', 'EC_mon']

# QC: print rows with num_obs > 11
rows_with_more_than_11_obs = EC_mon_df[EC_mon_df['num_obs'] > 11]
if not rows_with_more_than_11_obs.empty:
    print(rows_with_more_than_11_obs)
else:
    print("No rows with > 11 observations.")

# Save EC_mon_df as sheet 'mon'
with pd.ExcelWriter(os.path.join(other_obs_dir + 'IMPROVE_EC_Summary.xlsx'), engine='openpyxl', mode='w') as writer:
    EC_mon_df.to_excel(writer, sheet_name='mon', index=False)

# Calculate the annual average 'ECf_Val' for each 'SiteName', 'Latitude', 'Longitude'
EC_annual_df = EC_mon_df.groupby(['SiteName', 'Latitude', 'Longitude'])['EC_mon'].mean().reset_index()
EC_annual_df.columns = ['SiteName', 'Latitude', 'Longitude', 'EC_annual']

# Save the annual DataFrame as 'annual'
with pd.ExcelWriter(os.path.join(other_obs_dir + 'IMPROVE_EC_Summary.xlsx'), engine='openpyxl', mode='a') as writer:
    EC_annual_df.to_excel(writer, sheet_name='annual', index=False)


################################################################################################
# Other: Summarize EMEP EC data -1
################################################################################################

other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
EMEP_dir = other_obs_dir + '/EMEP_EC_2019_raw/'

# Initialize empty list to store extracted information
all_data = []

# Iterate through each file in the directory
for filename in os.listdir(EMEP_dir):
    # Exclude files starting with '.'
    if not filename.startswith('.'):
        if filename.endswith('.nas'):
            file_path = os.path.join(EMEP_dir, filename)
            with open(file_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()

                # Extract station name
                station_code = next(
                    (line.split(':')[1].strip() for line in lines if line.startswith('Station code:')), '')

                # Search for line starting with 'starttime'
                start_index = None
                for idx, line in enumerate(lines):
                    if line.startswith('starttime'):
                        start_index = idx
                        break

                # If 'starttime' line found
                if start_index is not None:
                    # Extract data from following lines
                    data = [line.split() for line in lines[start_index:]]
                    # Convert each entry to numeric except the line starting with 'starttime'
                    for i, row in enumerate(data):
                        if i != 0:  # Skip the line starting with 'starttime'
                            data[i] = [float(val) for val in row]  # Convert entries to float

                    # Append station name and data to all_data list
                    all_data.append((station_code, data))

# Save data into separate spreadsheets
with pd.ExcelWriter(os.path.join(other_obs_dir, 'EMEP_EC_raw.xlsx'), engine='openpyxl', mode='w') as writer:
    for station_code, data in all_data:
        # Convert data into DataFrame
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=station_code, index=False, header=False)

################################################################################################
# Other: Summarize EMEP EC data -2
################################################################################################

other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
EMEP_dir = other_obs_dir + '/EMEP_EC_2019_raw/'

# Initialize empty lists to store extracted information
EC_df = []

# Initialize counters
processed_files = 0

# Iterate through each file in the directory
for filename in os.listdir(EMEP_dir):
    # Exclude files starting with '.'
    if not filename.startswith('.'):
        if filename.endswith('.nas'):
            processed_files += 1  # Increment processed files counter
            file_path = os.path.join(EMEP_dir, filename)
            with open(file_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()

                # Extract required information
                station_code = next((line.split(':')[1].strip() for line in lines if line.startswith('Station code:')),"")
                station_name = next((line.split(':')[1].strip() for line in lines if line.startswith('Station name:')), "")
                latitude = next((line.split(':')[1].strip() for line in lines if line.startswith('Station latitude:')), "")
                longitude = next((line.split(':')[1].strip() for line in lines if line.startswith('Station longitude:')), "")
                component = next((line.split(':')[1].strip() for line in lines if line.startswith('Component:')), "")
                unit = next((line.split(':')[1].strip() for line in lines if line.startswith('Unit:')), "")
                analytical_measurement_technique = next((line.split(':')[1].strip() for line in lines if line.startswith('Analytical measurement technique:')), "")
                # print(station_name, latitude, longitude)

                # Append extracted information to EC_df list
                EC_df.append([station_code, station_name, latitude, longitude, component, unit, analytical_measurement_technique, filename])

# Convert the list of lists into a DataFrame
df = pd.DataFrame(EC_df, columns=['Station Code', 'Station Name', 'Latitude', 'Longitude', 'Component', 'Unit', 'Analytical_Measurement_Technique', 'Filename'])

# Save the DataFrame as an Excel file
with pd.ExcelWriter(os.path.join(other_obs_dir, 'EMEP_EC_Summary.xlsx'), engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, index=False)

# Print the counts
print(f"Processed files: {processed_files}")

################################################################################################
# Other: Combine measurement and GCHP dataset based on lat/lon
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

################################################################################################
# Other: Map SPARTAN and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(12, 5))
left = 0.03  # Adjust the left position
bottom = 0.01  # Adjust the bottom position
width = 0.94  # Adjust the width
height = 0.9  # Adjust the height
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 160, -60, 60], crs=ccrs.PlateCarree())
# ax.set_extent([70, 130, 20, 50], crs=ccrs.PlateCarree()) # China
# ax.set_extent([-130, -60, 15, 50], crs=ccrs.PlateCarree()) # US
# ax.set_extent([6, 25, 40, 60], crs=ccrs.PlateCarree()) # Europe

# Define the colormap
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]

# Create a custom colormap using the gradient defined
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)

vmax = 4  # 8 for BC, 150 for PM25, 15 for SO4, 0.25 for BC_PM25, 2 for BC_SO4

# Accumulate data for each face over the year
annual_v = None

for face in range(6):
    for mon in range(1, 13):
        sim_df = xr.open_dataset(
            sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon),
            engine='netcdf4') # CEDS
        x = sim_df.corner_lons.isel(nf=face)
        y = sim_df.corner_lats.isel(nf=face)
        v = sim_df[species].isel(nf=face)
        if annual_v is None:
            annual_v = v
        else:
            annual_v = annual_v + v

    # Calculate the annual average
    annual_v /= 12
    annual_v = annual_v.squeeze()
    print(x.shape, y.shape, annual_v.shape)

    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_v, cmap=cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_other_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim
print(compar_notna['source'].unique())

# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Observation
s2 = [120] * len(obs)  # outer ring: Simulation
markers = {'SPARTAN': 'o', 'other': 's'}

# Create scatter plot for other data points (squares)
for i, row in compar_notna.iterrows():
    source = row['source']
    if source != 'SPARTAN':  # Exclude SPARTAN data for now
        marker = markers.get(source, 'o')
        plt.scatter(x=row['lon'], y=row['lat'], c=row['obs'], s=s1[i], marker=marker, edgecolor='black',
                    linewidth=1, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=4)
        plt.scatter(x=row['lon'], y=row['lat'], c=row['sim'], s=s2[i], marker=marker, edgecolor='black',
                    linewidth=1, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=3)

# Create scatter plot for SPARTAN data points (circles)
for i, row in compar_notna.iterrows():
    source = row['source']
    if source == 'SPARTAN':  # Plot SPARTAN data
        marker = markers.get(source, 'o')
        plt.scatter(x=row['lon'], y=row['lat'], c=row['obs'], s=s1[i], marker=marker, edgecolor='black',
                    linewidth=1, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=4)
        plt.scatter(x=row['lon'], y=row['lat'], c=row['sim'], s=s2[i], marker=marker, edgecolor='black',
                    linewidth=1, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=3)

# Calculate the global mean of simulated and observed data
global_mean_sim = np.nanmean(sim)
global_mean_obs = np.nanmean(obs)
global_std_sim = np.nanstd(sim)
global_std_obs = np.nanstd(obs)

# Display statistics as text annotations on the plot
month_str = calendar.month_name[mon]
ax.text(0.4, 0.12, f'Sim = 1.77 ± 1.84 µg/m$^3$', fontsize=16, fontname='Arial', transform=ax.transAxes)
ax.text(0.4, 0.05, f'Obs = 3.19 ± 2.49 µg/m$^3$', fontsize=16, fontname='Arial', transform=ax.transAxes)
ax.text(0.9, 0.05, f'2019', fontsize=16, fontname='Arial', transform=ax.transAxes)
# plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial') # PM$_{{2.5}}$

# Create an inset axes for the color bar at the left middle of the plot
cbar_axes = inset_axes(ax,
                           width='2%',
                           height='50%',
                           bbox_to_anchor=(-0.95, -0.35, 1, 1),  # (x, y, width, height) relative to top-right corner
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
cbar = plt.colorbar(im, cax=cbar_axes, orientation="vertical")
font_properties = font_manager.FontProperties(family='Arial', size=14)
cbar.set_ticks([0, 1, 2, 3, 4], fontproperties=font_properties)
cbar.ax.set_ylabel(f'{species} (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
cbar.ax.tick_params(axis='y', labelsize=14)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

# plt.savefig(out_dir + 'Fig2_WorldMap_{}_{}_{}_Sim_vs_SPARTAN_{}_{}_AnnualMean.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()

################################################################################################
# Other: Create scatter plot, all black
################################################################################################
# Read the file
# compr_df = pd.read_csv(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_{}_{}07_MonMean.csv'.format(cres, inventory, deposition, species, year)))
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_Sim_vs_SPARTAN_other_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 2.4]
x_range_2 = [2.4, compr_df['obs'].max()]
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(7, 6))
# Create scatter plot
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, s=20, alpha=1, color='k', edgecolor='k', markers='o')
scatterplot.set_xscale('log')
scatterplot.set_yscale('log')
# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([0.005, 50]) # 11 for edgar
plt.ylim([0.005, 50])
# Set tick locations and labels manually
plt.xticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'], fontname='Arial', size=18)
plt.yticks([0.01, 0.1, 1, 10], ['0.01', '0.1', '1', '10'], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
plt.plot([compr_df['obs'].min(), compr_df['obs'].max()], [compr_df['obs'].min(), compr_df['obs'].max()], color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['obs'][mask], compr_df['sim'][mask])
# Plot regression lines
# sns.regplot(x='obs', y='sim', data=compr_df[mask], scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
plt.text(0.75, 0.05, f'{year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS4_Scatter_{}_{}_{}_Sim_vs_SPARTAN_other_{}_AnnualMean.svg'.format(cres, inventory, deposition, species), dpi=300)

plt.show()
################################################################################################
# SPARTAN HIPS vs UV-Vis vs IBR
################################################################################################
# Set the directory path
HIPS_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/'
IBR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'
################################################################################################
# Combine HIPS and UV-Vis dataset, set site as country and city
################################################################################################
# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Read the HIPS_df
HIPS_df = pd.read_excel(out_dir + 'BC_HIPS_SPARTAN.xlsx', sheet_name='All',
                        usecols=['FilterID', 'mass_ug', 'Volume_m3', 'BC_HIPS_ug', 'Site', 'Country', 'City'])
HIPS_count = HIPS_df.groupby('Site').size().reset_index(name='HIPS_count')

# Read the UV-Vis data from Analysis_ALL
UV_df = pd.read_excel(os.path.join(UV_dir, 'BC_UV-Vis_SPARTAN_Joshin_20230510.xlsx'), usecols=['Filter ID', 'f_BC', 'Location ID'])
# IBR_df = pd.read_excel(os.path.join(IBR_dir, 'BC_IBR_SPARTAN.xlsx'), usecols=['Sample ID#', 'EBC_ug'])
# Drop the last two digits in the "Filter ID" column: 'AEAZ-0113-1' to 'AEAZ-0113'
UV_df['FilterID'] = UV_df['Filter ID'].str[:-2]
UV_df.rename(columns={'Location ID': 'Site'}, inplace=True)
# Count the number of rows for each site
UV_count = UV_df.groupby('Site').size().reset_index(name='UV_count')

# Merge DataFrames
merged_df = pd.merge(UV_df, HIPS_df, on=['FilterID'], how='inner')

# Convert the relevant columns to numeric to handle any non-numeric values
merged_df['BC_HIPS_ug'] = pd.to_numeric(merged_df['BC_HIPS_ug'], errors='coerce')
merged_df['f_BC'] = pd.to_numeric(merged_df['f_BC'], errors='coerce')
merged_df['mass_ug'] = pd.to_numeric(merged_df['mass_ug'], errors='coerce')
merged_df['Volume_m3'] = pd.to_numeric(merged_df['Volume_m3'], errors='coerce')

# Calculate BC mass
merged_df['BC_UV-Vis_ug'] = merged_df['f_BC'] * merged_df['mass_ug']

# Calculate BC concentrations
merged_df['BC_HIPS_(ug/m3)'] = merged_df['BC_HIPS_ug'] / merged_df['Volume_m3']
merged_df['BC_UV-Vis_(ug/m3)'] = merged_df['f_BC'] * merged_df['mass_ug'] / merged_df['Volume_m3']
# merged_df['BC_IBR_(ug/m3)'] = merged_df['BC_IBR_ug'] / merged_df['Volume_m3']
# Calculate BC fractions
merged_df.rename(columns={'f_BC': 'f_BC_UV-Vis'}, inplace=True)
merged_df['f_BC_HIPS'] = merged_df['BC_HIPS_ug'] / merged_df['mass_ug']
# merged_df['f_BC_IBR'] = merged_df['BC_IBR_ug'] / merged_df['mass_ug']
# Drop rows where IBR BC < 0
# merged_df = merged_df.loc[merged_df['BC_IBR_ug'] > 0]

# Drop the "Site_y" column
merged_df.drop('Site_y', axis=1, inplace=True)
# Rename the "Site_x" column to "Site"
merged_df.rename(columns={'Site_x': 'Site'}, inplace=True)

# Write the merged data to separate sheets in an Excel file
with pd.ExcelWriter(os.path.join(out_dir, 'BC_HIPS_UV-Vis_SPARTAN.xlsx'), engine='openpyxl') as writer:
    # Write the merged data
    merged_df.to_excel(writer, sheet_name='HIPS_UV-Uis', index=False)


################################################################################################
# plot HIPS vs UV-Vis, color cell by no. of pairs
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(out_dir, 'BC_HIPS_UV-Vis_SPARTAN.xlsx'))
# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]
# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)
merged_df.rename(columns={"City": "city"}, inplace=True)

# Create a 2D histogram to divide the area into squares and count data points in each square
hist, xedges, yedges = np.histogram2d(merged_df['HIPS'], merged_df['UV-Vis'], bins=60)

# Determine the color for each square based on the number of pairs
colors = np.zeros_like(hist)
for i in range(len(hist)):
    for j in range(len(hist[i])):
        pairs = hist[i][j]
        colors[i][j] = pairs

# Define the custom color scheme gradient
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]

# Create a custom colormap using the gradient defined
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the 2D histogram with the specified color scheme
sns.set(font='Arial')
scatterplot = plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, origin='lower')

# Display the original data points as a scatter plot
# plt.scatter(merged_df['HIPS'], merged_df['IBR'], color='black', s=10, alpha=0.5)

# Set title, xlim, ylim, ticks, labels
plt.xlim([merged_df['HIPS'].min()-0.5, 17])
plt.ylim([merged_df['HIPS'].min()-0.5, 17])
plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = merged_df['HIPS']
y = merged_df['HIPS']
plt.plot([merged_df['HIPS'].min(), merged_df['HIPS'].max()], [merged_df['HIPS'].min(), merged_df['HIPS'].max()], color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=ax.transAxes, fontsize=18)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['HIPS']) & ~np.isnan(merged_df['UV-Vis'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['HIPS'][mask], merged_df['UV-Vis'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='HIPS', y='UV-Vis', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=18)

plt.xlabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('UV-Vis Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create the colorbar and specify font properties
cbar_ax = fig.add_axes([0.68, 0.25, 0.02, 0.4])
cbar = plt.colorbar(label='Number of Pairs', cax=cbar_ax)
cbar.ax.set_ylabel('Number of Pairs', fontsize=14, fontname='Arial')
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 5, 10, 15, 20], fontname='Arial', fontsize=14)

# show the plot
plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "BC_Comparison_HIPS_UV-Vis.svg"), format="SVG", dpi=300)
plt.show()

################################################################################################
# calculate IBR BC
################################################################################################
# Constants
q = 1.5 # thickness
filter_surface_area = 3.142  # cm^2
absorption_cross_section = 0.060  # cm^2/ug, 0.1 in SOP, 0.075 for AEAZ

# Define a function to calculate EBC (ug)
def calculate_ebc(row):
    normalized_r = row['normalized_r']
    if normalized_r > 0:  # Check if normalized_r is positive
        return  -filter_surface_area / (q * absorption_cross_section) * np.log(normalized_r / 1)
    else:
        return np.nan  # Replace invalid values with NaN

# Initialize an empty DataFrame to store the combined data
IBR_df = pd.DataFrame()

# Loop through each subfolder in the specified directory
for subfolder in os.listdir(IBR_dir):
    subfolder_path = os.path.join(IBR_dir, subfolder)
    # Check if the subfolder is a directory
    if os.path.isdir(subfolder_path):
        # Look for Excel files in the subfolder
        for file in os.listdir(subfolder_path):
            # Skip files starting with a dot
            if file.startswith('.') or file.startswith('~'):
                continue
            if file.endswith('_IBR.xlsx'):
                # Extract site name from the file name
                site_name = file.split('_')[0]
                print(site_name)
                # Read the Excel file
                excel_path = os.path.join(subfolder_path, file)
                df = pd.read_excel(excel_path, engine='openpyxl', header=7)
                # Extract required columns
                df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
                required_columns = ['Cartridge ID', 'Sample ID#', 'Reflectance']
                df = df[required_columns]
                # Drop rows with blank values in 'Reflectance' column
                df.dropna(subset=['Reflectance'], inplace=True)
                # Add 'site' column with site name
                df['site'] = site_name
                # Concatenate data with the combined DataFrame
                IBR_df = pd.concat([IBR_df, df])

# Write the merged data to separate sheets in an Excel file
with pd.ExcelWriter(os.path.join(out_dir, 'BC_IBR_SPARTAN.xlsx'),  engine='openpyxl', mode='w') as writer:
    # Write the merged data
    IBR_df.to_excel(writer, sheet_name='raw', index=False)

# # the normalized reflectance is wrong for most cartridges
# Normalize 'Reflectance' for each 'Cartridge ID'
# IBR_df = IBR_df.copy()
# IBR_df = pd.read_excel(os.path.join(out_dir, 'BC_IBR_SPARTAN.xlsx'))
# for cart_id, group in IBR_df.groupby('Cartridge ID'):
    # print(cart_id)
    # Find the 'Reflectance' value for 'Sample ID#' ending with '-7'
    # ref_7 = group.loc[group['Sample ID#'].str.endswith('-7'), 'Reflectance'].iloc[0]
    # Calculate normalized 'Reflectance' values for other 'Sample ID#'s
    # group['normalized_r'] = group['Reflectance'] / ref_7
    # Calculate EBC (ug) for each row
    # group['EBC_ug'] = group.apply(calculate_ebc, axis=1)
    # Update the DataFrame with the calculated EBC values
    # IBR_df.loc[group.index, 'normalized_r'] = group['normalized_r']
    # IBR_df.loc[group.index, 'EBC_ug'] = group['EBC_ug']
# Write the combined data with normalized 'Reflectance' to a new Excel file
# with pd.ExcelWriter(os.path.join(out_dir, 'BC_IBR_SPARTAN.xlsx'), engine='openpyxl', mode='a') as writer:
    # Write the merged data
    # IBR_df.to_excel(writer, sheet_name='IBR_EBC_py', index=False)
