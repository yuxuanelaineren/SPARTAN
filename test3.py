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
species = 'BC'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower())
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}/'.format(cres.lower(), species)
################################################################################################
# Create scatter plot for monthly and annual data
################################################################################################
# Read the file
# annual_df = pd.read_csv(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}01_MonMean.csv'.format(cres, species, year)))
# annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, species, year)), sheet_name='Annual')
annual_df = pd.read_excel(os.path.join(out_dir, '{}_LUO_Sim_vs_CAWNET_{}_{}_Summary.xlsx'.format(cres, species, year)), sheet_name='Annual')

# Drop rows where BC is greater than 1
# annual_df = annual_df.loc[annual_df['obs'] <= 20]

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
        (1, 0.42, 0.70), (0.8, 0.52, 0.7), (1, 0.48, 0.41), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ],  # Pink shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.2, 0.2), (1, 0.4, 0.4), (1, 0.6, 0.6)
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
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=10)
# legend.get_frame().set_linewidth(0.0)

# legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12)

# Set title, xlim, ylim, ticks, labels
plt.title(f'BC/(PM$_{{2.5}}$ - Nitrate) Comparison: GCHP-v13.4.1 {cres.lower()} v.s. SPARTAN',
          fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.01, 0.5])
plt.ylim([-0.01, 0.5])
# plt.xlim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
# plt.ylim([annual_df['sim'].min()-0.5, annual_df['sim'].max()+0.5])
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontname='Arial', size=18)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = annual_df['obs']
y = annual_df['obs']
plt.plot([annual_df['obs'].min(), annual_df['obs'].max()], [annual_df['obs'].min(), annual_df['obs'].max()],
         color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(annual_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=22)
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
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=22)

plt.xlabel('SPARTAN BC/(PM$_{{2.5}}$ - Nitrate)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('GCHP BC/(PM$_{{2.5}}$ - Nitrate)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_MonMean.tiff'.format(cres, species, year), dpi=600)
# plt.savefig(out_dir + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)
plt.savefig('/Users/renyuxuan/Downloads/' + 'Scatter_{}_Sim_vs_SPARTAN_{}_{:02d}_AnnualMean.tiff'.format(cres, species, year), dpi=600)

plt.show()
