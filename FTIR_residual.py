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
FTIR_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/' # old algrothrm
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/'
################################################################################################
# Match FTIR_OM, FTIR_OC, and Residual from SPARTAN
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(Residual_dir):
    # Define the desired column names
    desired_columns = ['Site_Code', 'Latitude', 'Longitude', 'Start_Year_local', 'Start_Month_local', 'Start_Day_local', 'Value', 'Flag',
                       'Filter PM2.5 mass', 'Fine Soil', 'OC PM2.5', 'kappa', 'Residual Matter'
                       ]
    Residual_dfs = []
    for filename in os.listdir(Residual_dir):
        if filename.endswith('.csv'):
            try:
                master_data = pd.read_csv(os.path.join(Residual_dir, filename), skiprows=3, encoding='ISO-8859-1',
                                          usecols=['Site_Code', 'Latitude', 'Longitude', 'Start_Year_local','Start_Month_local',
                                                   'Start_Day_local', 'Parameter_Name', 'Value', 'Flag'])
                # Create a pivot table to reshape the data so that each Parameter_Name becomes a column
                pivot_df = master_data.pivot_table(
                    index=['Site_Code', 'Latitude', 'Longitude', 'Start_Year_local', 'Start_Month_local', 'Start_Day_local'],
                    columns='Parameter_Name',
                    values='Value',
                    aggfunc='first'  # Use the first value if there are duplicates
                ).reset_index()
                # Keep only the desired columns
                pivot_df = pivot_df[[col for col in pivot_df.columns if col in desired_columns]]
                # Rename columns for clarity and consistency
                pivot_df.rename(
                    columns={
                        'Site_Code': 'Site',
                        'Filter PM2.5 mass': 'PM2.5',
                        'Fine Soil': 'Dust',
                        'OC PM2.5': 'OC_RCFM',
                        'kappa': 'Kappa',
                        'Residual Matter': 'Residual_RCFM'
                    },
                    inplace=True
                )

                # Combine date
                pivot_df['Date'] = pd.to_datetime(pivot_df['Start_Year_local'].astype(str) + '-' + pivot_df['Start_Month_local'].astype(str) + '-' + pivot_df['Start_Day_local'].astype(str))
                # Append the current HIPS_df to the list
                Residual_dfs.append(pivot_df)
            except Exception as e:
                print(f"Error occurred while processing file '{filename}': {e}. Skipping to the next file.")
    return pd.concat(Residual_dfs, ignore_index=True)

# Main script
if __name__ == '__main__':
    # Read data
    Residual_df = read_master_files(Residual_dir)
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City'])
    Residual_df = pd.merge(Residual_df, site_df, how="left", left_on="Site", right_on="Site_Code").drop("Site_Code", axis=1)
    OM_b4_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch4_2024_03', usecols=['site', 'date', 'OM', 'FTIR_OC'], skiprows=1)
    OM_b2_b3_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch2_v3_batch3_v2', usecols=['site', 'date', 'OM', 'FTIR_OC'], skiprows=1)
    OM_b1_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch1_2020_08', usecols=['Site', 'Date', 'aCOH', 'aCH', 'naCO', 'COOH', 'FTIR_OC'])
    OM_b4_df.rename(columns={'FTIR_OC': 'OC', 'site': 'Site', 'date': 'Date'}, inplace=True)
    OM_b2_b3_df.rename(columns={'FTIR_OC': 'OC', 'site': 'Site', 'date': 'Date'}, inplace=True)
    # Add a 'batch' column to each DataFrame
    OM_b4_df['batch'] = 'batch4_2024_03'
    OM_b2_b3_df['batch'] = 'batch2_2022_06_batch3_2023_03'
    OM_b1_df['batch'] = 'batch1_2020_08'
    OM_df = pd.concat([OM_b4_df, OM_b2_b3_df]) # exlcude batch 1 as no lot specific calibrations
    # Merge Residual and OM df based on matching values of "Site" and "Date"
    merged_df = pd.merge(Residual_df, OM_df, on=['Site', 'Date'], how='inner')
    merged_df.rename(columns={'Value': 'Residual'}, inplace=True)
    # merged_df.rename(columns={'Country': 'country'}, inplace=True)
    # merged_df.rename(columns={'City': 'city'}, inplace=True)
    # Write to Excel
    with pd.ExcelWriter(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_SPARTAN.xlsx'), engine='openpyxl', mode='w') as writer: # write mode
        merged_df.to_excel(writer, sheet_name='OM_OC_batch234_Residual', index=False)

################################################################################################
# Process RCFM from Chris and apppend date to filter
################################################################################################
# Function to read and preprocess RM data from Chris
def read_and_preprocess_rm_data(out_dir, filename='Updated_RCFM_Chris_raw.xlsx'):
    """
    Function to read and preprocess RM data from Chris.

    Args:
    - out_dir (str): Directory path where the Excel file is located.
    - filename (str): Name of the Excel file to read (default is 'Updated_RCFM_Chris.xlsx').

    Returns:
    - RM_df (DataFrame): A concatenated DataFrame with data from all sheets and a 'SheetName' column.
    """
    RM_df = pd.DataFrame()
    RM_data = pd.read_excel(os.path.join(out_dir, filename), sheet_name=None)
    sheet_names = RM_data.keys()
    # Loop through each sheet and append to RM_df with a new column for sheet name
    for sheet in sheet_names:
        # Read the sheet into a dataframe
        df_sheet = RM_data[sheet]
        df_sheet['Site'] = sheet  # Add a 'SheetName' column
        RM_df = pd.concat([RM_df, df_sheet], ignore_index=True)
    return RM_df
# Function to read and preprocess master data
def read_master_files(SPARTAN_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350', 'MXMC-0006', 'NGIL-0309'
    ]
    SPARTAN_dfs = []
    for filename in os.listdir(SPARTAN_dir):
        if filename.endswith('.csv'):
            master_data = pd.read_csv(os.path.join(SPARTAN_dir, filename), encoding='ISO-8859-1')
            SPARTAN_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'mass_ug', 'Volume_m3']
            if all(col in master_data.columns for col in SPARTAN_columns):
                master_data = master_data[SPARTAN_columns]
                master_data['mass_ug'] = pd.to_numeric(master_data['mass_ug'], errors='coerce')
                master_data['Volume_m3'] = pd.to_numeric(master_data['Volume_m3'], errors='coerce')
                master_data['PM2.5_master'] = master_data['mass_ug'] / master_data['Volume_m3']
                # Convert to string and remove whitespace
                master_data['start_year'] = master_data['start_year'].astype(str).str.strip()
                # Convert columns to numeric, filling invalid values with NaN, and then replace NaNs with 0 or a valid default
                master_data['start_year'] = pd.to_numeric(master_data['start_year'], errors='coerce', downcast='integer')
                # master_data['Date'] = pd.to_datetime(master_data['start_year'].astype(str) + '-' + master_data['start_month'].astype(str) + '-' + master_data['start_day'].astype(str))
                SPARTAN_dfs.append(master_data)
            else:
                print(f"Skipping {filename} because not all required columns are present.")
    return pd.concat(SPARTAN_dfs, ignore_index=True)

# Main script
if __name__ == '__main__':
    # Read RM data from Chris
    RM_df = read_and_preprocess_rm_data(out_dir)
    SPARTAN_df = read_master_files(SPARTAN_dir)
    RM_df = RM_df.merge(SPARTAN_df[['FilterID', 'start_year', 'start_month', 'start_day', 'PM2.5_master']], on='FilterID', how='left')
    with pd.ExcelWriter(os.path.join(out_dir, "Updated_RCFM_Chris_Summary.xlsx"), engine='openpyxl', mode='w') as writer:
        RM_df.to_excel(writer, sheet_name='All', index=False)
    # Read FT-IR OM
    OM_b4_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch4_2024_03',
                             usecols=['site', 'date', 'OM', 'FTIR_OC'], skiprows=1)
    OM_b2_b3_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch2_v3_batch3_v2',
                                usecols=['site', 'date', 'OM', 'FTIR_OC'], skiprows=1)
    OM_b1_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch1_2020_08',
                             usecols=['Site', 'Date', 'aCOH', 'aCH', 'naCO', 'COOH', 'FTIR_OC'])
    OM_b4_df.rename(columns={'FTIR_OC': 'OC', 'site': 'Site', 'date': 'Date'}, inplace=True)
    OM_b2_b3_df.rename(columns={'FTIR_OC': 'OC', 'site': 'Site', 'date': 'Date'}, inplace=True)
    # Add a 'batch' column to each DataFrame
    OM_b4_df['batch'] = 'batch4_2024_03'
    OM_b2_b3_df['batch'] = 'batch2_2022_06_batch3_2023_03'
    OM_b1_df['batch'] = 'batch1_2020_08'
    OM_df = pd.concat([OM_b4_df, OM_b2_b3_df])  # exlcude batch 1 as no lot specific calibrations
    # Merge RM and OM df based on matching values of "Site" and "Date"
    RM_df['start_year'] = RM_df['start_year'].astype(int)
    RM_df['start_month'] = RM_df['start_month'].astype(int)
    RM_df['start_day'] = RM_df['start_day'].astype(int)
    RM_df['Date'] = pd.to_datetime(RM_df['start_year'].astype(str) + '-' +RM_df['start_month'].astype(str) + '-' + RM_df['start_day'].astype(str))

    merged_df = pd.merge(RM_df, OM_df, on=['Site', 'Date'], how='inner')
    # merged_df.rename(columns={'Country': 'country'}, inplace=True)
    # merged_df.rename(columns={'City': 'city'}, inplace=True)
    # Write to Excel
    with pd.ExcelWriter(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_Chris.xlsx'), engine='openpyxl',
                        mode='w') as writer:  # write mode
        merged_df.to_excel(writer, sheet_name='OM_OC_batch234_Residual', index=False)

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
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_SPARTAN.xlsx'), sheet_name='OM_OC_20_22new_23_Residual')
compr_df = compr_df[compr_df['OM'] > 0]
compr_df = compr_df[compr_df['batch'].isin(['2022_06_new', '2023_03'])]
# compr_df['Ratio'] = compr_df['OM'] / compr_df['FTIR_OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['FTIR_OC']*2.5, axis=1)

# Print the names of each city
unique_cities = compr_df['City'].unique()
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
        (0, 0, 0.6),  (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9), (0.85, 0.9, 1)
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

# Calculate mean and standard error grouped by City
summary_stats = compr_df.groupby('City').agg(
    OM_mean=('OM', 'mean'),
    OM_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
    Residual_mean=('Residual', 'mean'),
    Residual_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='OM', y='Residual', data=compr_df, hue='City', palette=city_palette, s=20, alpha=1, edgecolor='k', style='City',  markers=city_marker)
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
plt.title('FT-IR OM (Batch 2 and 3) vs Residual', fontsize=18, fontname='Arial', y=1.03)
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
plt.plot([-5, 50], [-5, 50], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [compr_df['OM'].min(), compr_df['OM'].max()]
# Perform linear regression for all segments
mask = (compr_df['OM'] >= x_range[0]) & (compr_df['OM'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['OM'][mask], compr_df['Residual'][mask])
# Plot regression lines
sns.regplot(x='OM', y='Residual', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add columns for normalized mean difference (NMD) and normalized root mean square difference (NRMSD)
def calculate_nmd_and_nrmsd(df, obs_col, sim_col):
    """
    Calculate normalized mean difference (NMD) and normalized root mean square difference (NRMSD).

    Args:
        df (pd.DataFrame): DataFrame containing observation and simulation columns.
        obs_col (str): Column name for observations.
        sim_col (str): Column name for simulations.

    Returns:
        dict: Dictionary containing NMD and NRMSD values.
    """
    obs = df[obs_col].values
    sim = df[sim_col].values
    # Remove rows with NaN values
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]
    # Check if there are valid data points
    if len(obs) == 0:
        return {'NMD (%)': np.nan, 'NRMSD (%)': np.nan}
    # Calculate NMD
    nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMD (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='OM', sim_col='Residual')
nmd = nmd_nrmsd_results['NMD (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.70, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}\nNMD = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=16, color='black')

# Set labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_B23_vs_Residual.svg', dpi=300)

plt.show()

################################################################################################
# Create scatter plot for Residual vs OM, colored by dust
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
def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city)
            assigned_marker = region_markers[region][city_index % len(region_markers[region])]
            return assigned_marker
    return None
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_Chris.xlsx'), sheet_name='OM_OC_batch234_Residual')
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df.rename(columns={'RM_dry': 'Residual'}, inplace=True)
compr_df['DustRatio'] = compr_df['Soil'] / compr_df['PM2.5']
compr_df['Residual'] = compr_df.apply(lambda row: row['Residual'] if row['DustRatio'] < 0.4 else row['Residual'] + row['PM2.5']*(row['DustRatio'] - 0.4), axis=1)
# compr_df = compr_df[compr_df['OM'] > 0]
# compr_df = compr_df[compr_df['OC'] > 0]
# compr_df = compr_df[compr_df['Residual'] > 0]
# compr_df['Ratio'] = compr_df['OM'] / compr_df['OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['OC']*2.5, axis=1)

# Step 1: Calculate monthly mean and standard error for each city
monthly_stats = compr_df.groupby(['City', 'start_month']).agg(
    OM_monthly_mean=('OM', 'mean'),
    OM_monthly_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
    Residual_monthly_mean=('Residual', 'mean'),
    Residual_monthly_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Step 2: Calculate annual statistics (mean and standard error) from monthly results
annual_stats = monthly_stats.groupby(['City']).agg(
    OM_mean=('OM_monthly_mean', 'mean'),
    OM_se=('OM_monthly_se', 'mean'),
    Residual_mean=('Residual_monthly_mean', 'mean'),
    Residual_se=('Residual_monthly_se', 'mean')
).reset_index()
# Rename the annual statistics DataFrame as summary_df and sort by OM_mean
summary_df = annual_stats.sort_values(by='OM_mean')

# Print the names of each city
unique_cities = compr_df['City'].unique()
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

# Create a colormap for DustRatio
blue_colors = [(0, 0, 0.6), (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
colors = blue_colors + [(1, 1, 1)] + red_colors
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
# nnorm = plt.Normalize(vmin=0, vmax=0.4)
norm = plt.Normalize(vmin=compr_df['DustRatio'].min(), vmax=compr_df['DustRatio'].max())

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='OM', y='Residual', data=compr_df, hue='DustRatio', palette=cmap, s=20, alpha=1, edgecolor='k', style='City',  markers=city_marker)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)

# Sort the legend labels based on x-axis order
handles, labels = ax.get_legend_handles_labels()
ordered_handles_labels = sorted(zip(summary_df['OM_mean'], handles, labels), key=lambda x: x[0])
_, ordered_handles, ordered_labels = zip(*ordered_handles_labels)

# Add the legend with ordered labels
legend = ax.legend(ordered_handles, ordered_labels, markerscale=0.7, prop={'family': 'Arial','size': 11.5}, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('FT-IR OM vs updated Residual, Dust Fraction < 0.4', fontsize=18, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-9, 80])
plt.ylim([-9, 80])
plt.xticks([0, 20, 40, 60, 80], fontname='Arial', size=18)
plt.yticks([0, 20, 40, 60, 80], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['OM']
y = compr_df['OM']
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [compr_df['OM'].min(), compr_df['OM'].max()]
# Perform linear regression for all segments
mask = (compr_df['OM'] >= x_range[0]) & (compr_df['OM'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['OM'][mask], compr_df['Residual'][mask])
# Plot regression lines
sns.regplot(x='OM', y='Residual', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add columns for normalized mean difference (NMD) and normalized root mean square difference (NRMSD)
def calculate_nmd_and_nrmsd(df, obs_col, sim_col):
    """
    Calculate normalized mean difference (NMD) and normalized root mean square difference (NRMSD).

    Args:
        df (pd.DataFrame): DataFrame containing observation and simulation columns.
        obs_col (str): Column name for observations.
        sim_col (str): Column name for simulations.

    Returns:
        dict: Dictionary containing NMD and NRMSD values.
    """
    obs = df[obs_col].values
    sim = df[sim_col].values
    # Remove rows with NaN values
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]
    # Check if there are valid data points
    if len(obs) == 0:
        return {'NMD (%)': np.nan, 'NRMSD (%)': np.nan}
    # Calculate NMD
    nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMD (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='OM', sim_col='Residual')
nmd = nmd_nrmsd_results['NMD (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.65, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}\nNMD = {nmd:.1f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Set labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create an axis for the colorbar (cax)
cax = fig.add_axes([0.63, 0.2, 0.015, 0.3])  # Position: [x, y, width, height]
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=cax,  # Set the colorbar to the specified axis
)
cbar.set_label('Dust Fraction', fontsize=14, fontname='Arial', labelpad=5)
cbar.ax.tick_params(labelsize=12, width=1)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1], fontname='Arial', fontsize=14)

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_B234_vs_updatedResidual_ColorByDust<0.4.svg', dpi=300)

plt.show()

################################################################################################
# Create scatter plot for Residual vs OM, mean+se, colored by region
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
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_Chris.xlsx'), sheet_name='OM_OC_batch234_Residual')
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df.rename(columns={'RM_dry': 'Residual'}, inplace=True)
# compr_df = compr_df[compr_df['OM'] > 0]
# compr_df = compr_df[compr_df['OC'] > 0]
# compr_df = compr_df[compr_df['Residual'] > 0]
compr_df['Ratio'] = compr_df['OM'] / compr_df['OC']
compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['OC']*2.5, axis=1)

# Print the names of each city
unique_cities = compr_df['City'].unique()
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
        (0, 0, 0.6),  (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9), (0.85, 0.9, 1)
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

# Step 1: Calculate monthly mean and standard error for each city
monthly_stats = compr_df.groupby(['City', 'start_month']).agg(
    OM_monthly_mean=('OM', 'mean'),
    OM_monthly_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
    Residual_monthly_mean=('Residual', 'mean'),
    Residual_monthly_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Step 2: Calculate annual statistics (mean and standard error) from monthly results
annual_stats = monthly_stats.groupby(['City']).agg(
    OM_mean=('OM_monthly_mean', 'mean'),
    OM_se=('OM_monthly_se', 'mean'),
    Residual_mean=('Residual_monthly_mean', 'mean'),
    Residual_se=('Residual_monthly_se', 'mean')
).reset_index()

# Rename the annual statistics DataFrame as summary_df and sort by OM_mean
summary_df = annual_stats.sort_values(by='OM_mean')

# # Calculate mean and standard error grouped by City
# summary_df = compr_df.groupby('City').agg(
#     OM_mean=('OM', 'mean'),
#     OM_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
#     Residual_mean=('Residual', 'mean'),
#     Residual_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
# ).reset_index()
# summary_df = summary_df.sort_values(by='OM_mean')

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# Create scatter plot with white background, black border, and no grid
sns.set(style="whitegrid", font="Arial", font_scale=1.2)
# Iterate through each city to plot individual points and error bars
for _, row in summary_df.iterrows():
    city = row['City']
    color = map_city_to_color(city)  # Get color for the city
    marker = map_city_to_marker(city)  # Get marker for the city
    # Plot the mean values with error bars
    scatterplot = ax.errorbar(
        row['OM_mean'], row['Residual_mean'],
        xerr=row['OM_se'], yerr=row['Residual_se'],
        fmt=marker, color=color, ecolor='black', elinewidth=1, capsize=3, label=city, markersize=8,
        markeredgecolor='black', markeredgewidth=0.5,
    )
border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid

# Sort the legend labels based on x-axis order
handles, labels = ax.get_legend_handles_labels()
ordered_handles_labels = sorted(zip(summary_df['OM_mean'], handles, labels), key=lambda x: x[0])
_, ordered_handles, ordered_labels = zip(*ordered_handles_labels)

# Add the legend with ordered labels
legend = ax.legend(ordered_handles, ordered_labels, markerscale=0.7, prop={'family': 'Arial','size': 11.5}, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('FT-IR OM vs updated Residual, OM/OC < 2.5', fontsize=16, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-3, 35])
plt.ylim([-3, 35])
plt.xticks([0, 10, 20, 30], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = summary_df['OM_mean']
y = summary_df['OM_mean']
plt.plot([-5, 50], [-5, 50], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [summary_df['OM_mean'].min(), summary_df['OM_mean'].max()]
# Perform linear regression for all segments
mask = (summary_df['OM_mean'] >= x_range[0]) & (summary_df['OM_mean'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(summary_df['OM_mean'][mask], summary_df['Residual_mean'][mask])
# Plot regression lines
sns.regplot(x='OM_mean', y='Residual_mean', data=summary_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add columns for normalized mean difference (NMD) and normalized root mean square difference (NRMSD)
def calculate_nmd_and_nrmsd(df, obs_col, sim_col):
    """
    Calculate normalized mean difference (NMD) and normalized root mean square difference (NRMSD).

    Args:
        df (pd.DataFrame): DataFrame containing observation and simulation columns.
        obs_col (str): Column name for observations.
        sim_col (str): Column name for simulations.

    Returns:
        dict: Dictionary containing NMD and NRMSD values.
    """
    obs = df[obs_col].values
    sim = df[sim_col].values
    # Remove rows with NaN values
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]
    # Check if there are valid data points
    if len(obs) == 0:
        return {'NMD (%)': np.nan, 'NRMSD (%)': np.nan}
    # Calculate NMD
    nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMD (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(summary_df, obs_col='OM_mean', sim_col='Residual_mean')
nmd = nmd_nrmsd_results['NMD (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.65, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}\n$N$ = {num_points}\nNMD = {nmd:.1f}%\nNRMSD = {nrmsd:.0f}%',
         transform=ax.transAxes, fontsize=18, color='black')

# Set labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_B234_vs_updatedResidual_AnnualMean_OMOC2.5.svg', dpi=300)

plt.show()
################################################################################################
# Create scatter plot for Residual vs OM, mean+se, colored by dust fraction
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
def map_city_to_marker(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city)
            assigned_marker = region_markers[region][city_index % len(region_markers[region])]
            return assigned_marker
    return None

# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_Residual_Chris.xlsx'), sheet_name='OM_OC_batch234_Residual')
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df.rename(columns={'RM_dry': 'Residual'}, inplace=True)
compr_df['DustRatio'] = compr_df['Soil'] / compr_df['PM2.5']
compr_df['Residual'] = compr_df.apply(lambda row: row['Residual'] if row['DustRatio'] < 0.4 else row['Residual'] + row['PM2.5']*(row['DustRatio'] - 0.4), axis=1)
# compr_df = compr_df[compr_df['OM'] > 0]
# compr_df = compr_df[compr_df['OC'] > 0]
# compr_df = compr_df[compr_df['Residual'] > 0]
# compr_df['Ratio'] = compr_df['OM'] / compr_df['OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['OC']*2.5, axis=1)

# Step 1: Calculate monthly mean and standard error for each city
monthly_stats = compr_df.groupby(['City', 'start_month']).agg(
    OM_monthly_mean=('OM', 'mean'),
    DustRatio_monthly_mean=('DustRatio', 'mean'),
    OM_monthly_se=('OM', lambda x: x.std() / np.sqrt(len(x))),
    Residual_monthly_mean=('Residual', 'mean'),
    Residual_monthly_se=('Residual', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Step 2: Calculate annual statistics (mean and standard error) from monthly results
annual_stats = monthly_stats.groupby(['City']).agg(
    OM_mean=('OM_monthly_mean', 'mean'),
    DustRatio_mean=('DustRatio_monthly_mean', 'mean'),
    OM_se=('OM_monthly_se', 'mean'),
    Residual_mean=('Residual_monthly_mean', 'mean'),
    Residual_se=('Residual_monthly_se', 'mean')
).reset_index()
# Calculate the range of DustRatio_mean
dustratio_min = annual_stats['DustRatio_mean'].min()
dustratio_max = annual_stats['DustRatio_mean'].max()
# Print the range
print(f"Range of DustRatio_mean: {dustratio_min} to {dustratio_max}")

# Rename the annual statistics DataFrame as summary_df and sort by OM_mean
summary_df = annual_stats.sort_values(by='OM_mean')

# Print the names of each city
unique_cities = compr_df['City'].unique()
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

# Create a colormap for DustRatio
blue_colors = [(0, 0, 0.6), (0, 0, 1), (0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
colors = blue_colors + [(1, 1, 1)] + red_colors
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
norm = plt.Normalize(vmin=0, vmax=0.4)
# norm = plt.Normalize(vmin=summary_df['DustRatio_mean'].min(), vmax=summary_df['DustRatio_mean'].max())
# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# Create scatter plot with white background, black border, and no grid
sns.set(style="whitegrid", font="Arial", font_scale=1.2)

# Iterate through each city to plot individual points and error bars
for _, row in summary_df.iterrows():
    city = row['City']
    color = cmap(norm(compr_df.loc[compr_df['City'] == city, 'DustRatio'].values[0]))  # Get color for the city based on DustRatio
    marker = map_city_to_marker(city)  # Get marker for the city
    # Plot the mean values with error bars
    scatterplot = ax.errorbar(
        row['OM_mean'], row['Residual_mean'],
        xerr=row['OM_se'], yerr=row['Residual_se'],
        fmt=marker, color=color, ecolor='black', elinewidth=1, capsize=3, label=city, markersize=8,
        markeredgecolor='black', markeredgewidth=0.5,
    )
border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid

# Sort the legend labels based on x-axis order
handles, labels = ax.get_legend_handles_labels()
ordered_handles_labels = sorted(zip(summary_df['OM_mean'], handles, labels), key=lambda x: x[0])
_, ordered_handles, ordered_labels = zip(*ordered_handles_labels)

# Add the legend with ordered labels
legend = ax.legend(ordered_handles, ordered_labels, markerscale=0.7, prop={'family': 'Arial','size': 11.5}, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('FT-IR OM vs updated Residual, Dust Fraction < 0.4', fontsize=16, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-3, 35])
plt.ylim([-3, 35])
plt.xticks([0, 10, 20, 30], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = summary_df['OM_mean']
y = summary_df['OM_mean']
plt.plot([-5, 50], [-5, 50], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [summary_df['OM_mean'].min(), summary_df['OM_mean'].max()]
# Perform linear regression for all segments
mask = (summary_df['OM_mean'] >= x_range[0]) & (summary_df['OM_mean'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(summary_df['OM_mean'][mask], summary_df['Residual_mean'][mask])
# Plot regression lines
sns.regplot(x='OM_mean', y='Residual_mean', data=summary_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add columns for normalized mean difference (NMD) and normalized root mean square difference (NRMSD)
def calculate_nmd_and_nrmsd(df, obs_col, sim_col):
    """
    Calculate normalized mean difference (NMD) and normalized root mean square difference (NRMSD).

    Args:
        df (pd.DataFrame): DataFrame containing observation and simulation columns.
        obs_col (str): Column name for observations.
        sim_col (str): Column name for simulations.

    Returns:
        dict: Dictionary containing NMD and NRMSD values.
    """
    obs = df[obs_col].values
    sim = df[sim_col].values
    # Remove rows with NaN values
    valid_indices = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid_indices]
    sim = sim[valid_indices]
    # Check if there are valid data points
    if len(obs) == 0:
        return {'NMD (%)': np.nan, 'NRMSD (%)': np.nan}
    # Calculate NMD
    nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMD (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(summary_df, obs_col='OM_mean', sim_col='Residual_mean')
nmd = nmd_nrmsd_results['NMD (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.65, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}\n$N$ = {num_points}\nNMD = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=ax.transAxes, fontsize=18, color='black')

# Set labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Add colorbar to the plot
# cbar = fig.colorbar(
#     plt.cm.ScalarMappable(cmap=cmap, norm=norm),
#     ax=ax,  # Specify which axis the colorbar should be linked to
#     # orientation='vertical',  # Vertical colorbar
#     # fraction=0.03,  # Fraction of the parent axis to use
#     # pad=-0.15,  # Padding between the colorbar and the y-axis
#     # shrink=0.5  # Shrink the height to 70% of its original size
# )
# # cbar.ax.set_position([0.28, 0.05, 0.02, 0.4])  # [x, y, width, height]
# cbar.set_label('Dust Fraction', fontsize=12, fontname='Arial', labelpad=10)
# cbar.set_ticks([0, 0.2, 0.4])
# cbar.ax.tick_params(labelsize=10, width=1.5)
# cbar.outline.set_edgecolor('black')
# cbar.outline.set_linewidth(1)
# cbar.ax.set_position([0.28, 0.05, 0.02, 0.4])  # [x, y, width, height]


# Create an axis for the colorbar (cax)
cax = fig.add_axes([0.63, 0.2, 0.015, 0.3])  # Position: [x, y, width, height]
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=cax,  # Set the colorbar to the specified axis
)
cbar.set_label('Dust Fraction', fontsize=14, fontname='Arial', labelpad=5)
cbar.ax.tick_params(labelsize=12, width=1)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 0.1, 0.2, 0.3, 0.4], fontname='Arial', fontsize=14)

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'OM_B234_vs_updatedResidual_AnnualMean_ColorByDust<0.4.svg', dpi=300)

plt.show()