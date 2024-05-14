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

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/'
################################################################################################
# Match FTIR_OM, FTIR_OC, and Residual
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(Residual_dir):
    Residual_dfs = []
    for filename in os.listdir(Residual_dir):
        if filename.endswith('.csv'):
            try:
                master_data = pd.read_csv(os.path.join(Residual_dir, filename), skiprows=3, encoding='ISO-8859-1',
                                          usecols=['Site_Code', 'Latitude', 'Longitude', 'Start_Year_local','Start_Month_local',
                                                   'Start_Day_local', 'Parameter_Name', 'Value', 'Flag'])
                # Select Residual
                Residual_df = master_data.loc[master_data['Parameter_Name'] == 'Residual Matter'].copy()
                Residual_df.rename(columns={'Site_Code': 'Site'}, inplace=True)
                # Combine date
                Residual_df['Date'] = pd.to_datetime(Residual_df['Start_Year_local'].astype(str) + '-' + Residual_df['Start_Month_local'].astype(str) + '-' + Residual_df['Start_Day_local'].astype(str))
                # Append the current HIPS_df to the list
                Residual_dfs.append(Residual_df)
            except Exception as e:
                print(f"Error occurred while processing file '{filename}': {e}. Skipping to the next file.")
    return pd.concat(Residual_dfs, ignore_index=True)

# Main script
if __name__ == '__main__':
    # Read data
    Residual_df = read_master_files(Residual_dir)
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City'])
    Residual_df = pd.merge(Residual_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)
    OM_23_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2023_03', usecols=['Site', 'Date', 'OM', 'FTIR_OC'])
    OM_22_new_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2022_06_new', usecols=['Site', 'Date', 'OM', 'FTIR_OC'])
    OM_20_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_raw_all_20230506.xlsx'), sheet_name='2020_09', usecols=['Site', 'Date', 'OM', 'OC'])
    OM_23_df.rename(columns={'FTIR_OC': 'OC'}, inplace=True)
    OM_22_new_df.rename(columns={'FTIR_OC': 'OC'}, inplace=True)
    OM_df = pd.concat([OM_20_df, OM_22_new_df, OM_23_df])
    # Merge Residual and OM df based on matching values of "Site" and "Date"
    merged_df = pd.merge(Residual_df, OM_df, on=['Site', 'Date'], how='inner')
    merged_df.rename(columns={'Value': 'Residual'}, inplace=True)
    # merged_df.rename(columns={'Country': 'country'}, inplace=True)
    # merged_df.rename(columns={'City': 'city'}, inplace=True)
    # Write to Excel
    with pd.ExcelWriter(os.path.join(out_dir, 'OM_OC_Residual_SPARTAN.xlsx'), engine='openpyxl', mode='a') as writer:
        merged_df.to_excel(writer, sheet_name='OM_OC_Residual_20_22new_23', index=False)
    with pd.ExcelWriter(os.path.join(out_dir, 'OM_Residual_SPARTAN.xlsx'), engine='openpyxl', mode='a') as writer:
        merged_df.to_excel(writer, sheet_name='OM_Residual_20_22new_23', index=False)

################################################################################################
# Create scatter plot for Residual vs OM, colored by region
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
compr_df = pd.read_excel(os.path.join(out_dir, 'OM_Residual_SPARTAN.xlsx'), sheet_name='OM_Residual')
# compr_df['Ratio'] = compr_df['OM'] / compr_df['FTIR_OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['FTIR_OC']*2.5, axis=1)

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
x_range = [compr_df['OM'].min(), compr_df['OM'].max()]
# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='OM', y='Residual', data=compr_df, hue='city', palette=city_palette, s=20, alpha=1, edgecolor='k', style='city',  markers=city_marker)
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
        handle = plt.Line2D([0], [0], marker=marker, color=color, linestyle='', markersize=6, label=city)
        legend_handles.append(handle)

# Create legend with custom handles
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('Batch 2 and 3: FT-IR OM vs Residual', fontsize=18, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-5, 48])
plt.ylim([-5, 48])
plt.xticks([0, 10, 20, 30, 40], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30, 40], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['OM']
y = compr_df['OM']
plt.plot([compr_df['OM'].min(), 50], [compr_df['OM'].min(), 50], color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['OM'] >= x_range[0]) & (compr_df['OM'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['OM'][mask], compr_df['Residual'][mask])
# Plot regression lines
sns.regplot(x='OM', y='Residual', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.1, 0.76, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
# plt.text(0.66, 0.05, f'Batch 2 and 3', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_vs_Residual_site_2.5_threstold.svg', dpi=300)

plt.show()

################################################################################################
# Plot FTIR_OM and Residual, colored by no. of pairs
################################################################################################
# Read the file
merged_df = pd.read_excel(os.path.join(out_dir, 'OM_Residual_SPARTAN.xlsx'), sheet_name='OM_Residual_20_22new_23')
merged_df = merged_df.loc[merged_df['Residual'] < 50]
# merged_df['Ratio'] =merged_df['OM'] / merged_df['FTIR_OC']
# merged_df['OM'] = merged_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['FTIR_OC']*2.5, axis=1)

# Create a 2D histogram to divide the area into squares and count data points in each square
hist, xedges, yedges = np.histogram2d(merged_df['OM'], merged_df['Residual'], bins=60)

# Determine the color for each square based on the number of pairs
colors = np.zeros_like(hist)
for i in range(len(hist)):
    for j in range(len(hist[i])):
        pairs = hist[i][j]
        colors[i][j] = pairs

# Define the custom color scheme gradient
colors = [(1, 1, 1),(0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]

# Create a custom colormap using the gradient defined
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the 2D histogram with the specified color scheme
sns.set(font='Arial')
scatterplot = plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, origin='lower')

# Display the original data points as a scatter plot
# plt.scatter(merged_df['OM'], merged_df['Residual'], color='black', s=10, alpha=0.5)

# Set title, xlim, ylim, ticks, labels
plt.title('Batch 1, 2 and 3: FT-IR OM vs Residual', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-5, 48])
plt.ylim([-5, 48])
plt.xticks([0, 10, 20, 30, 40], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30, 40], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
plt.plot([merged_df['Residual'].min(), merged_df['Residual'].max()], [merged_df['Residual'].min(), merged_df['Residual'].max()], color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=ax.transAxes, fontsize=18)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['OM']) & ~np.isnan(merged_df['Residual'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['OM'][mask], merged_df['Residual'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='OM', y='Residual', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=18)

plt.xlabel('FT-IR Orgainc Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create the colorbar and specify font properties
cbar_ax = fig.add_axes([0.68, 0.58, 0.02, 0.3])
# cbar_ax = fig.add_axes([0.72, 0.20, 0.02, 0.3])
cbar = plt.colorbar(label='Number of Pairs', cax=cbar_ax)
cbar.ax.set_ylabel('Number of Pairs', fontsize=14, fontname='Arial')
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 10, 20, 30, 40], fontname='Arial', fontsize=14)
ax.set_aspect(0.9 / 1)
# show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_vs_Residual_pairs_20_22new_23.svg', dpi=300)
plt.show()

################################################################################################
# Combine FTIR OC and GCHP OM/OC based on lat/lon and seasons
################################################################################################

# Load data
sim_df = xr.open_dataset(OMOC_dir + 'OMOC.DJF.01x01.nc', engine='netcdf4') # DJF, JJA, MAM, SON
obs_df = pd.read_excel(out_dir + 'OM_OC_Residual_SPARTAN.xlsx', sheet_name='OM_OC_Residual_20_22new_23')
site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Country', 'City', 'Latitude', 'Longitude'])

# Define a function to map months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'JJA'
    elif month in [6, 7, 8]:
        return 'MAM'
    elif month in [9, 10, 11]:
        return 'SON'
    else:
        return 'Unknown'

# Add a new column 'season' based on 'Start_Month_local'
obs_df.rename(columns={'Start_Month_local': 'month'}, inplace=True)
obs_df['season'] = obs_df['month'].apply(get_season)

# Extract lon/lat from SPARTAN site
site_lon = site_df['Longitude']
site_df.loc[site_df['Longitude'] > 180, 'Longitude'] -= 360
site_lat = site_df['Latitude']

# Extract lon/lat, and OMOC from sim
sim_lon = np.array(sim_df.coords['lon']) # Length of sim_lon: 3600
sim_lon[sim_lon > 180] -= 360
sim_lat = np.array(sim_df.coords['lat']) # Length of sim_lat: 1800
sim_conc = np.array(sim_df['OMOC'])

# Initialize lists to store data
sim_data = []
site_data = []

# Iterate over each site
for site_index, (site_lon, site_lat) in enumerate(zip(site_lon, site_lat)):
    # Find the nearest simulation latitude and longitude to the site
    sim_lat_nearest = sim_df.lat.sel(lat=site_lat, method='nearest').values
    sim_lon_nearest = sim_df.lon.sel(lon=site_lon, method='nearest').values

    # Extract the corresponding simulation concentration
    sim_conc_nearest = sim_df.sel(lat=sim_lat_nearest, lon=sim_lon_nearest)['OMOC'].values

    # Append the data to the lists
    sim_data.append((sim_lat_nearest, sim_lon_nearest, sim_conc_nearest))
    site_data.append((site_lat, site_lon))

# Create DataFrame with simulation and site data
sim_site_df = pd.DataFrame(sim_data, columns=['sim_lat', 'sim_lon', 'sim_OMOC'])
sim_site_df['site_lat'] = [data[0] for data in site_data]
sim_site_df['site_lon'] = [data[1] for data in site_data]

# Merge site_df with sim_site_df based on latitude and longitude
sim_site_df = pd.merge(sim_site_df, site_df[['Latitude', 'Longitude', 'Country', 'City']],
                       left_on=['site_lat', 'site_lon'], right_on=['Latitude', 'Longitude'], how='left')
# Drop the redundant latitude and longitude columns from site_df
sim_site_df.drop(columns=['Latitude', 'Longitude'], inplace=True)

# Print the resulting DataFrame
print(sim_site_df)

with pd.ExcelWriter(out_dir + 'OMOC_SPARTAN_Summary.xlsx', engine='openpyxl') as writer:
    sim_site_df.to_excel(writer, sheet_name='DJF', index=False)
