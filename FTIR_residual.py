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
import matplotlib.dates as mdates

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
            city_index = cities.index(city)
            assigned_marker = region_markers[region][city_index % len(region_markers[region])]
            return assigned_marker
    return None
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx'), sheet_name='All')
compr_df['Residual'] = compr_df['RM_dry']
compr_df['OM'] = compr_df['FTIR_OM']
compr_df = compr_df[compr_df['OM'] > 0]
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df = compr_df[compr_df['Site'].isin(['AEAZ', 'ILHA', 'ILNZ'])]
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
    # 'Central Asia': [
    #     (0.58, 0.1, 0.81), (0.66, 0.33, 0.83), (0.9, 0.4, 1), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    # ],  # Purple shades
    'Central Asia': [
        'black', 'blue','red'
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
plt.title('FT-IR OM vs updated Residual', fontsize=18, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-8, 52])
plt.ylim([-8, 52])
plt.xticks([0, 10, 20, 30, 40, 50], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial', size=18)
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
# # Plot regression lines
# sns.regplot(x='OM', y='Residual', data=compr_df[mask],
#             scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

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
    # nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    nmd = np.sum(sim - obs) / np.sum(obs) * 100
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMB (%)': nmd, 'NRMSD (%)': nrmsd}
# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='OM', sim_col='Residual')
nmd = nmd_nrmsd_results['NMB (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.70, f'y = {slope:.3f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.3f}\nN = {num_points}\nNMB = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=16, color='black')

# Set labels
plt.xlabel('FT-IR OM (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FTIR_OM_vs_Residual_AbuDhabi_Haifa_Rehovot.svg', dpi=300)

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
# Create scatter plot for Residual vs OM vs GEOS-Chem OM/OC, mean+se, colored by region
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
def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx'), sheet_name='All')
compr_df = compr_df[compr_df['batch'].isin(['batch2_2022_06_batch3_2023_03', 'batch4_2024_03'])]
compr_df.rename(columns={'RM_dry': 'Residual'}, inplace=True)
# If City is Abu Dhabi, set Residual = RM_dry + PBW; otherwise, keep Residual = RM_dry
# compr_df['Residual'] = compr_df.apply(lambda row: row['Residual'] + row['PBW'] if row['City'] == 'Abu Dhabi' else row['Residual'], axis=1)
# compr_df['OM'] = compr_df['FTIR_OC'] * compr_df['sim_OMOC']
compr_df['OM'] = compr_df['FTIR_OM']
# compr_df = compr_df[compr_df['OM'] > 0]
# compr_df = compr_df[compr_df['OC'] > 0]
# compr_df = compr_df[compr_df['Residual'] > 0]
# compr_df['Ratio'] = compr_df['OM'] / compr_df['OC']
# compr_df['OM'] = compr_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['OC']*2.5, axis=1)

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
plt.title('FT-IR OM vs updated Residual', fontsize=16, fontname='Arial', y=1.03)
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
    # nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    nmd = np.sum(sim - obs) / np.sum(obs) * 100
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
# plt.xlabel('FT-IR OC × GEOS-Chem OM/OC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.xlabel('FT-IR OM (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Residual (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FTIR_OM_vs_updatedResidual_AEAZwaterToResidual.jpg', dpi=300)

plt.show()
################################################################################################
# Extract GCHP OM/OC at SPARTAN sites
################################################################################################
# Load data
site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Country', 'City', 'Latitude', 'Longitude'])

# Create an empty DataFrame to store all seasons
all_seasons_df = pd.DataFrame()
# Loop through each season file
for season_file in ['OMOC.DJF.01x01.nc', 'OMOC.JJA.01x01.nc', 'OMOC.MAM.01x01.nc', 'OMOC.SON.01x01.nc']:
    # Load data
    sim_df = xr.open_dataset(OMOC_dir + season_file, engine='netcdf4')

    # Add a new column 'season' based on the current file being processed
    season = season_file.split('.')[1]
    sim_df['season'] = season

    # Extract lon/lat from SPARTAN site
    site_lon = site_df['Longitude']
    site_df.loc[site_df['Longitude'] > 180, 'Longitude'] -= 360
    site_lat = site_df['Latitude']

    # Extract lon/lat, and OMOC from sim
    sim_lon = np.array(sim_df.coords['lon'])
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.coords['lat'])
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
        sim_conc_nearest = sim_df.sel(lat=sim_lat_nearest, lon=sim_lon_nearest)['OMOC'].values[0]

        # Append the data to the lists
        sim_data.append((sim_lat_nearest, sim_lon_nearest, sim_conc_nearest))
        site_data.append((site_lat, site_lon))

        print(site_index)

    # Create DataFrame with sim and site data
    sim_site_df = pd.DataFrame(sim_data, columns=['sim_lat', 'sim_lon', 'sim_OMOC'])
    sim_site_df['site_lat'] = [data[0] for data in site_data]
    sim_site_df['site_lon'] = [data[1] for data in site_data]

    # Merge site_df with sim_site_df based on latitude and longitude
    sim_site_df = pd.merge(sim_site_df, site_df[['Latitude', 'Longitude', 'Country', 'City']],
                           left_on=['site_lat', 'site_lon'], right_on=['Latitude', 'Longitude'], how='left')
    # Add a 'season' column
    sim_site_df['season'] = season
    # Drop the redundant latitude and longitude columns from site_df
    sim_site_df.drop(columns=['Latitude', 'Longitude'], inplace=True)

    # Concatenate the current season's DataFrame with all_seasons_df
    all_seasons_df = pd.concat([all_seasons_df, sim_site_df], ignore_index=True)

# Print the resulting DataFrame
print(all_seasons_df)

# Write the summary DataFrame to an Excel file
with pd.ExcelWriter(out_dir + 'sim_OMOC_at_SPARTAN_site.xlsx', engine='openpyxl', mode='w') as writer:
    all_seasons_df.to_excel(writer, sheet_name='All_Seasons', index=False)

################################################################################################
# Combine FTIR OC and extracted GCHP OM/OC based on lat/lon and seasons
################################################################################################
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

# Load data
obs_df = pd.read_excel(out_dir + 'FT-IR_OM_OC_Residual_Chris.xlsx', sheet_name='OM_OC_batch234_Residual')
sim_df = pd.read_excel(out_dir + 'sim_OMOC_at_SPARTAN_site.xlsx', sheet_name='All_Seasons')

# Add a new column 'season' based on 'Start_Month_local'
obs_df.rename(columns={'start_month': 'month'}, inplace=True)
obs_df['season'] = obs_df['month'].apply(get_season)
obs_df.rename(columns={'OM': 'FTIR_OM', 'OC': 'FTIR_OC'}, inplace=True)

# Merge obs_df and sim_df based on 'season', 'Latitude', and 'Longitude' in obs_df, and 'season', 'site_lat', and 'site_lon' in sim_df
sim_obs_df = pd.merge(obs_df, sim_df, left_on=['season', 'Country', 'City'], right_on=['season', 'Country', 'City'], how='inner')

# Drop the redundant latitude and longitude columns from site_df
# sim_obs_df.drop(columns=['sim_lat', 'sim_lon'], inplace=True)

print('sim_obs_df:', sim_obs_df)

# Write the summary DataFrame to an Excel file
with pd.ExcelWriter(out_dir + 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx', engine='openpyxl', mode='w') as writer:
    sim_obs_df.to_excel(writer, sheet_name='All', index=False)

################################################################################################
# Create scatter plot for collocation comparison
################################################################################################

# Define a function to calculate the perpendicular distance from a point to the 1:1 line (y=x)
def perpendicular_distance(x, y):
    return np.abs(y - x) / np.sqrt(2)
def calculate_precision(x, y):
    x, y = np.array(x), np.array(y)  # Ensure arrays
    valid_mask = ~np.isnan(x) & ~np.isnan(y)  # Mask to remove NaNs
    x, y = x[valid_mask], y[valid_mask]  # Filter valid values

    if len(x) == 0 or len(y) == 0:  # If no valid pairs remain, return NaN
        return np.nan

    mean_xy = (x + y) / 2
    diff_squared = (x - y) ** 2
    precision = np.sqrt(np.mean(diff_squared / (2 * mean_xy ** 2)))
    print(f"{species_name}: {precision}")
    # print(np.isnan(x_values).sum(), np.isnan(y_values).sum())

    return precision

# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx'), sheet_name='All')
compr_df = compr_df[compr_df['Co-location'].notna()]
compr_df = compr_df[compr_df['City'] == 'Dhaka']
# Select pairs of rows with the same 'Co-location' value
pairs_df = compr_df[compr_df['Co-location'].isin([1, 26])]
compr_df['PM2.5 × 0.1'] = compr_df['PM2.5'] * 0.1
compr_df['PBW × 10'] = compr_df['PBW'] * 10

# Initialize the figure for plotting
fig, ax = plt.subplots(figsize=(8, 6))
# Set global font to Arial, size 18
plt.rcParams.update({'font.family': 'Arial', 'font.size': 18})

# Add 1:1 line with grey dash
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1, zorder=0)

# Define the species columns for plotting
species = ['PM2.5 × 0.1', 'BC', 'TEO', 'Soil', 'Na', 'NO3', 'NH4', 'SO4', 'PBW × 10', 'Cl', 'K', 'RM_dry', 'FTIR_OM']
species_colors = {
    'BC': 'black',
    'SO4': 'red',
    'NH4': 'pink',
    'NO3': 'orange',
    'PBW × 10': 'blue',
    'RM_dry': 'lightgreen',
    'FTIR_OM': 'darkgreen',
    'TEO': 'grey',
    'Soil': 'yellow',
    'Na': 'darkblue',
    'PM2.5 × 0.1': 'purple',  # Example color for PM2.5 * 0.1
    'Cl': 'cyan',  # Example color for Cl
    'K': 'brown'   # Example color for K
}
# Get unique species names in the dataset
unique_species = np.unique(species)
# Track plotted species to avoid duplicate legend entries
plotted_species = set()

# Create pairs based on 'Co-location' column
for co_location in compr_df['Co-location'].unique():
    pair = compr_df[compr_df['Co-location'] == co_location]

    if len(pair) == 2:  # Ensure there are exactly two rows for this co-location
        # Plot each species in the pair
        for species_name in species:
            x_value = pair.iloc[0][species_name]
            y_value = pair.iloc[1][species_name]

            # Determine marker style based on species name
            if species_name in ['RM_dry', 'FTIR_OM', 'PBW × 10']:
                marker_style = 's'  # square marker
                alpha_value = 1  # Full opacity for RM_dry and FTIR_OM
            else:
                marker_style = 'o'  # circle marker
                alpha_value = 0.8  # 80% opacity for other species
            # Get the color for the species
            color = species_colors.get(species_name, 'black')  # Default to black if species not found
            # Add label to the legend only the first time for each species
            if species_name not in plotted_species:
                scatterplot = ax.scatter(x_value, y_value, label=species_name, alpha=alpha_value, marker=marker_style, color=color,
                           edgecolor='k', zorder=2)
                plotted_species.add(species_name)
            else:
                scatterplot = ax.scatter(x_value, y_value, alpha=alpha_value, marker=marker_style, color=color, edgecolor='k', zorder=2)
# Set the border width of the x and y axes
ax.spines['top'].set_linewidth(1)  # Set top border width
ax.spines['right'].set_linewidth(1)  # Set right border width
ax.spines['bottom'].set_linewidth(1)  # Set bottom border width
ax.spines['left'].set_linewidth(1)  # Set left border width

# Sort the legend labels based on x-axis order
handles, labels = ax.get_legend_handles_labels()
# Add the legend with ordered labels
legend = ax.legend(handles, labels, markerscale=1, prop={'family': 'Arial','size': 12}, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
legend.get_frame().set_edgecolor('black')

# Define the species to check
species_to_check = ['RM_dry', 'FTIR_OM', 'PBW × 10']
# Initialize a dictionary to store precision for each species
precision_values = {}

# Iterate through the species and compute precision
for species_name in species:
    x_values = []
    y_values = []

    for co_location in compr_df['Co-location'].unique():
        pair = compr_df[compr_df['Co-location'] == co_location]
        if len(pair) == 2:  # Ensure exactly two rows per co-location
            x_values.append(pair.iloc[0][species_name])
            y_values.append(pair.iloc[1][species_name])

    # Compute precision only if there are valid pairs
    if len(x_values) > 0 and len(y_values) > 0:
        precision_values[species_name] = calculate_precision(x_values, y_values)
# Display precision values on the plot
y_offset = 0.97  # Starting position for displaying text
for species_name, precision in precision_values.items():
    ax.text(0.97, y_offset, f'Precision ({species_name}): {precision:.2f}',
            transform=ax.transAxes, fontsize=14, ha='right', va='top')
    y_offset -= 0.05  # Adjust spacing between lines

# for species_name, mean_error in mean_errors.items():
#     # Place the text at different y-positions to avoid overlap
#     ax.text(0.95, y_offset, f'Mean Error for {species_name}: {mean_error:.2f}',
#             transform=ax.transAxes, fontsize=18, ha='right', va='bottom')
#
#     # Increment the y_offset to place the next label on the next line
#     y_offset -= 0.05  # Adjust the spacing if necessary

# Count the number of pairs
num_pairs = len(compr_df['Co-location'].unique())
# Add the number of pairs to the plot
ax.text(0.97, 0.01, f'Number of Pairs: {num_pairs}', transform=ax.transAxes, fontsize=16, ha='right', va='bottom')

# Set plot labels and title
plt.xlim([-0.5, 30])
plt.ylim([-0.5, 30])
# plt.xticks([0, 1, 2, 3, 4, 5], fontname='Arial', size=18)
# plt.yticks([0, 1, 2, 3, 4, 5], fontname='Arial', size=18)
plt.xticks([0, 10, 20, 30], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)
plt.xlabel('First Co-location (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Second Co-location (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.title('Concentrations for Co-located Pairs', fontname='Arial', size=18, y=1.03)

# Display the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Co-location_Halifax.svg', dpi=300)

plt.show()
################################################################################################
# Create scatter plot for collocation comparison
################################################################################################
# Define a function to calculate the perpendicular distance from a point to the 1:1 line (y=x)
def perpendicular_distance(x, y): # scale dependent
    return np.abs(y - x) / np.sqrt(2)
def calculate_precision(x, y):
    x, y = np.array(x), np.array(y)  # Convert inputs to numpy arrays
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[valid_mask], y[valid_mask]
    n = len(x)  # Number of valid pairs
    if n == 0:  # If no valid pairs remain, return NaN
        return np.nan
    numerator = ((x - y) ** 2) / 2
    denominator = ((x + y) / 2) ** 2
    sigma_total = (1 / n) * np.sum(numerator / denominator)  # Apply formula
    precision =np.sqrt((1 / n) * np.sum(numerator / denominator))
    return precision

# Function to calculate NMB
def calculate_nmd(x, y):
    x, y = np.array(x), np.array(y)  # Ensure arrays
    valid_mask = ~np.isnan(x) & ~np.isnan(y)  # Mask to remove NaNs
    x, y = x[valid_mask], y[valid_mask]  # Filter valid values

    if len(x) == 0 or len(y) == 0:  # If no valid pairs remain, return NaN
        return np.nan

    # nmb = np.sum(x - y) / np.sum(y) * 100
    nmd = np.sum(np.abs(x - y)) / np.sum(y) * 100
    return nmd
# Function to calculate NRMSB
def calculate_nrmsb(x, y):
    x, y = np.array(x), np.array(y)  # Ensure arrays
    valid_mask = ~np.isnan(x) & ~np.isnan(y)  # Mask to remove NaNs
    x, y = x[valid_mask], y[valid_mask]  # Filter valid values

    if len(x) == 0 or len(y) == 0:  # If no valid pairs remain, return NaN
        return np.nan

    nrmsb = np.sqrt(np.mean((x - y) ** 2)) / np.sum(y) * 100
    return nrmsb

# Read the file
compr_df = pd.read_excel(os.path.join(Colocation_dir, 'Colocation_master_CAHA_BDDU.xlsx'), sheet_name='RCFM')
compr_df = compr_df[compr_df['Co-location'].notna()]
compr_df = compr_df[compr_df['City'] == 'Dhaka']
# Select pairs of rows with the same 'Co-location' value
pairs_df = compr_df[compr_df['Co-location'].isin([1, 26])]

# Define the columns that need to be converted (all _XRF_ng columns)
XRF_columns = [col for col in compr_df.columns if col.endswith('_XRF_ng')]
# Perform conversion for each of these columns
for col in XRF_columns:
    # Convert mass to concentration (ng/m3 to µg/m³)
    compr_df[f'{col.replace("_XRF_ng", "")}'] = compr_df[col] / compr_df['Volume_m3'] / 1000  # Divide by 1000 to convert ng to µg
compr_df['Soil × 0.1'] = compr_df['Soil'] * 0.1
compr_df['TEO × 0.05'] = compr_df['TEO'] * 0.05
compr_df['Zn × 0.1'] = compr_df['Zn'] * 0.1
compr_df['Pb × 0.1'] = compr_df['Pb'] * 0.1
compr_df['Si × 0.1'] = compr_df['Si'] * 0.1

# # Define the formula for the diff calculation
# def calculate_diff(row):
#     return row['TEO'] - (
#         1.79 * row['V'] +
#         1.69 * row['Cr'] +
#         1.63 * row['Mn'] +
#         1.34 * row['Co'] +
#         1.27 * row['Ni'] +
#         1.25 * row['Cu'] +
#         1.24 * row['Zn'] +
#         1.43 * row['As'] +
#         1.41 * row['Se'] +
#         1.09 * row['Rb'] +
#         1.18 * row['Sr'] +
#         1.14 * row['Cd'] +
#         1.20 * row['Sn'] +
#         1.26 * row['Sb'] +
#         1.20 * row['Ce'] +
#         1.12 * row['Pb']
#     )
# # Apply the calculation to each row and store the result in a new column 'diff'
# compr_df['diff'] = compr_df.apply(calculate_diff, axis=1)
# # Print FilterID along with the calculated diff
# for index, row in compr_df.iterrows():
#     print(f'FilterID: {row["FilterID"]}, diff: {row["diff"]:.2f}')

# Initialize the figure for plotting
fig, ax = plt.subplots(figsize=(8, 6))
# Set global font to Arial, size 18
plt.rcParams.update({'font.family': 'Arial', 'font.size': 18})

# Add 1:1 line with grey dash
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1, zorder=0)

# Define the species columns for plotting
species = ['Soil × 0.1', 'TEO × 0.05', 'Al', 'Si × 0.1', 'Ca', 'Fe', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni',
           'Cu', 'Zn × 0.1', 'As', 'Se', 'Rb', 'Sr', 'Cd', 'Sn', 'Sb', 'Ce', 'Pb × 0.1']
# Define species colors for plotting
species_colors = {
    'TEO × 0.05': 'grey',
    'Soil × 0.1': 'yellow',
    'Al': 'black',
    'Si × 0.1': 'red',
    'Ca': 'pink',
    'Fe': 'orange',
    'Ti': 'blue',
    'V': 'violet',
    'Cr': 'limegreen',
    'Mn': 'magenta',
    'Co': 'chocolate',
    'Ni': 'gold',
    'Cu': 'silver',
    'Zn × 0.1': 'teal',
    'As': 'plum',
    'Se': 'coral',
    'Rb': 'turquoise',
    'Sr': 'tomato',
    'Cd': 'orchid',
    'Sn': 'indigo',
    'Sb': 'lightseagreen',
    'Ce': 'darkslategray',
    'Pb × 0.1': 'darkred'
}

# Get unique species names in the dataset
unique_species = np.unique(species)
# Track plotted species to avoid duplicate legend entries
plotted_species = set()

# Create pairs based on 'Co-location' column
for co_location in compr_df['Co-location'].unique():
    pair = compr_df[compr_df['Co-location'] == co_location]

    if len(pair) == 2:  # Ensure there are exactly two rows for this co-location
        # Plot each species in the pair
        for species_name in species:
            x_value = pair.iloc[0][species_name]
            y_value = pair.iloc[1][species_name]

            # Determine marker style based on species name
            if species_name in ['Soil × 0.1', 'Al', 'Si × 0.1', 'Ca', 'Fe', 'Ti']:
                marker_style = 's'  # square marker
                alpha_value = 1  # Full opacity for RM_dry and FTIR_OM
            else:
                marker_style = 'o'  # circle marker
                alpha_value = 0.8  # 80% opacity for other species
            # Get the color for the species
            color = species_colors.get(species_name, 'black')  # Default to black if species not found
            # Add label to the legend only the first time for each species
            if species_name not in plotted_species:
                scatterplot = ax.scatter(x_value, y_value, label=species_name, alpha=alpha_value, marker=marker_style, color=color,
                           edgecolor='k', zorder=2)
                plotted_species.add(species_name)
            else:
                scatterplot = ax.scatter(x_value, y_value, alpha=alpha_value, marker=marker_style, color=color, edgecolor='k', zorder=2)
# Set the border width of the x and y axes
ax.spines['top'].set_linewidth(1)  # Set top border width
ax.spines['right'].set_linewidth(1)  # Set right border width
ax.spines['bottom'].set_linewidth(1)  # Set bottom border width
ax.spines['left'].set_linewidth(1)  # Set left border width

# Sort the legend labels based on x-axis order
handles, labels = ax.get_legend_handles_labels()
# Add the legend with ordered labels
legend = ax.legend(handles, labels, markerscale=1, prop={'family': 'Arial','size': 11}, loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
legend.get_frame().set_edgecolor('black')

# Initialize a dictionary to store NMD values for each species
nmd_values = {}

# Iterate through the species and compute NMD
for species_name in species:
    # Check if species_name contains "*0.1" or "*0.01"
    if ' × 0.1' in species_name:
        base_species_name = species_name.replace(' × 0.1', '')  # Remove '*0.1'
        multiply_factor = 1
    elif ' × 0.01' in species_name:
        base_species_name = species_name.replace(' × 0.01', '')  # Remove '*0.01'
        multiply_factor = 1
    elif ' × 0.05' in species_name:
        base_species_name = species_name.replace(' × 0.05', '')  # Remove '*0.01'
        multiply_factor = 1
    else:
        base_species_name = species_name  # No change if no multiplication factor
        multiply_factor = 1  # No multiplication factor for base species

    # Initialize lists to store x and y values for each species
    x_values = []
    y_values = []

    for co_location in compr_df['Co-location'].unique():
        pair = compr_df[compr_df['Co-location'] == co_location]
        if len(pair) == 2:  # Ensure exactly two rows per co-location
            # Adjust values by the multiplication factor if necessary
            x_values.append(pair.iloc[0][species_name] * multiply_factor)
            y_values.append(pair.iloc[1][species_name] * multiply_factor)

    # Compute NMD only if there are valid pairs
    if len(x_values) > 0 and len(y_values) > 0:
        nmd_values[base_species_name] = calculate_precision(x_values, y_values)

# Print out the results with base species names (without *0.1 or *0.01)
for species_name in nmd_values:
    print(f'{species_name} - calculate_precision: {nmd_values[species_name]:.2f}')


# Count the number of pairs
num_pairs = len(compr_df['Co-location'].unique())
# Add the number of pairs to the plot
ax.text(0.97, 0.01, f'Number of Pairs: {num_pairs}', transform=ax.transAxes, fontsize=16, ha='right', va='bottom')

# Set plot labels and title
plt.xlim([-0.01, 0.09])
plt.ylim([-0.01, 0.09])
plt.xticks([0, 0.03, 0.06, 0.09], fontname='Arial', size=16)
plt.yticks([0, 0.03, 0.06, 0.09], fontname='Arial', size=16)
# plt.xticks([0, 10, 20, 30], fontname='Arial', size=18)
# plt.yticks([0, 10, 20, 30], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)
plt.xlabel('First Co-location (µg/m$^3$)', fontsize=16, color='black', fontname='Arial')
plt.ylabel('Second Co-location (µg/m$^3$)', fontsize=16, color='black', fontname='Arial')
plt.title('Concentrations for Co-located Pairs: Halifax', fontname='Arial', size=16, y=1.03)

# Display the plot
plt.tight_layout()
# plt.savefig(Colocation_dir + 'Co-location_Dust_TEO_Halifax.svg', dpi=300)

plt.show()

################################################################################################
# Create bar plot and pir charts for precision
################################################################################################
# Data for Dhaka and Halifax
species = ['Soil', 'Al', 'Si', 'Ca', 'Fe', 'Ti', 'TEO', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Rb', 'Sr', 'Cd', 'Sn', 'Sb', 'Ce', 'Pb']
precision_dhaka = [0.39, 0.35, 0.45, 0.22, 0.36, 0.40, 0.40, 0.51, 0.90, 0.47, 0.57, 2.04, 1.51, 0.46, 0.47, 0.68, 0.44, 0.52, 1.42, 32.95, 0.61, 0.44, 0.55]
precision_halifax = [0.17, 0.19, 0.18, 0.50, 0.22, 1.61, 1.67, 0.20, 0.49, 0.26, 1.48, 1.26, 48.48, 0.36, 0.14, 0.20, 3.31, 1.52, 2.54, 2.26, 1.53, 0.36, 0.46]

# Modify the precision values for 'Cu' and 'Sn' by multiplying by 0.1
cu_index = species.index('Cu')
Sn_index = species.index('Sn')
precision_dhaka[cu_index] *= 0.1
precision_halifax[cu_index] *= 0.1
precision_dhaka[Sn_index] *= 0.1
precision_halifax[Sn_index] *= 0.1

# Set up the bar width and positions
bar_width = 0.35
index = np.arange(len(species))

# Create the plot
fig, ax = plt.subplots(figsize=(8, 9))

# Plotting bars for Dhaka and Halifax
bar1 = ax.barh(index, precision_dhaka, bar_width, label='Dhaka', color='b')
bar2 = ax.barh(index + bar_width, precision_halifax, bar_width, label='Halifax', color='r')
border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid
# Adjust the species label for 'Cu' and 'Sn' to reflect the adjusted precision
species[cu_index] = 'Cu × 0.1'
species[Sn_index] = 'Sn × 0.1'

# Add a grey dashed line before "TEO"
teo_index = species.index('TEO')
ax.axhline(y=teo_index - bar_width / 2, color='grey', linestyle='--', linewidth=1)

# Adding labels, title, and ticks
ax.set_ylabel('Species', fontsize=16, family='Arial')
ax.set_xlabel('Precision', fontsize=16, family='Arial')
ax.set_title('Precision for Dhaka and Halifax', fontsize=16, family='Arial')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(species, fontsize=16, family='Arial')
plt.xlim([0, 5])
plt.xticks([0, 1, 2, 3, 4, 5], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)
# Reverse the y-axis order
ax.invert_yaxis()

# Set the legend with correct font properties
legend=plt.legend(fontsize=18, prop={'family': 'Arial'})
legend.get_frame().set_edgecolor('white')

# Display the plot
plt.tight_layout(pad=2)
Colocation_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/Co-location/'
# plt.savefig(Colocation_dir + 'Co-location_Precision_Dust_TEO.svg', dpi=300)

plt.show()

################################################################################################
# Create stack bar plot and pir charts for Abu Dhabi
################################################################################################
# Define color mapping for each species
species_colors = {
    'BC': 'black',
    'SO4': '#ED3029',  # Red
    'NH4': '#F067A6',  # Pink
    'NO3': '#F57E20',  # Orange
    'PBW': '#6DCFF6',  # Light Blue
    'RM_dry': '#50B848',  # Green
    'FTIR_OM': 'darkgreen',
    'TEO': '#808285',  # Grey
    'Soil': '#FCEE1E',  # Yellow
    'Na': '#3954A5',  # Dark Blue
    'PM2.5': 'purple',
    'Cl': 'cyan',
    'K': 'brown'
}

# Read the file
file_path = os.path.join(out_dir, 'FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx')
compr_df = pd.read_excel(file_path, sheet_name='All')

# Filter data for Abu Dhabi
abu_dhabi_df = compr_df[compr_df['City'] == 'Abu Dhabi'].copy()

# Convert Date column to datetime format
abu_dhabi_df.loc[:, 'Date'] = pd.to_datetime(abu_dhabi_df['Date'])

# Select species for the stacked bar plot
columns_to_plot = ['BC', 'TEO', 'Soil', 'Na', 'NO3', 'NH4', 'SO4', 'PBW', 'Cl', 'K']
abu_dhabi_df[columns_to_plot] = abu_dhabi_df[columns_to_plot].clip(lower=0)

# Convert dates to numerical format for better width control
dates_num = mdates.date2num(abu_dhabi_df['Date'])
bar_width = 7  # Adjust for wider bars

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Add a grey dashed line at y = 0
ax.axhline(0, color='grey', linestyle='dashed', linewidth=1)

# Plot RM_dry below zero for negative values
ax.bar(dates_num, abu_dhabi_df['RM_dry'].clip(upper=0), width=bar_width,
       color=species_colors['RM_dry'], align='center', label='RM_dry')

# Start stacking from RM_dry (positive values)
bottom_values = abu_dhabi_df['RM_dry'].clip(lower=0)

# Plot the positive RM_dry explicitly so it's visible
ax.bar(dates_num, bottom_values, width=bar_width, color=species_colors['RM_dry'], align='center')

# Stack the other species on top of RM_dry
for specie in columns_to_plot:
    if specie in species_colors:
        ax.bar(dates_num, abu_dhabi_df[specie], width=bar_width, label=specie,
               color=species_colors[specie], bottom=bottom_values, align='center')
        bottom_values += abu_dhabi_df[specie]

# Format x-axis
ax.set_xlabel('Date', fontsize=16, family='Arial')
ax.set_ylabel('Concentration (µg/m$^3$)', fontsize=16, family='Arial')
ax.set_title('Stacked Bar Plot of Species Concentrations: Abu Dhabi', fontsize=18, family='Arial')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format x-axis labels
plt.xticks(rotation=45, fontsize=12, family='Arial')
plt.yticks(fontsize=12, family='Arial') # [10, 10, 20, 30]
plt.ylim([-10, 70])

# Add legend (only show RM_dry once)
handles, labels = ax.get_legend_handles_labels()
unique_legend = dict(zip(labels, handles))  # Remove duplicate labels
legend = ax.legend(unique_legend.values(), unique_legend.keys(), prop={'family': 'Arial','size': 12}, bbox_to_anchor=(1.01, 1), loc='upper left')
legend.get_frame().set_edgecolor('black')

# Set font for all texts in the plot
# plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})


# Save the plot as a JPG with 300 DPI
output_path = os.path.join(out_dir, 'Stacked_Bar_AbuDhabi.jpg')
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='jpg')

# Show plot
plt.tight_layout()
plt.show()
# # Filter data for Abu Dhabi
# abu_dhabi_df = compr_df[compr_df['City'] == 'Abu Dhabi']
# # Select the relevant columns for averaging
# columns_to_average = ['BC', 'TEO', 'Soil', 'Na', 'NO3', 'NH4', 'SO4', 'PBW', 'Cl', 'K', 'RM_dry']
# # Compute the average values
# average_values = abu_dhabi_df[columns_to_average].mean()
# # Remove negative values
# average_values = average_values[average_values > 0]
# # Check if there are valid values left
# if not average_values.empty:
#     # Get colors for available species
#     colors = [species_colors[specie] for specie in average_values.index]
#
#     # Plot the pie chart
#     plt.figure(figsize=(8, 8))
#     plt.pie(average_values, labels=average_values.index, autopct='%1.1f%%', startangle=140,
#             colors=colors, wedgeprops={'edgecolor': 'black'})
#
#     # Set title
#     plt.title('Average Composition in Abu Dhabi')
#
#     # Show the plot
#     plt.show()
# else:
#     print("No valid positive values to plot.")
################################################################################################
# Combine SPARTAN and GCHP dataset based on lat/lon
################################################################################################
cres = 'C360'
year = 2019
species = 'OA'
inventory = 'CEDS'
deposition = 'noLUO'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)

# Function to find matching rows and add 'Country' and 'City'
def find_and_add_location(lat, lon):
    for index, row in site_df.iterrows():
        if abs(row['Latitude'] - lat) <= 0.3 and abs(row['Longitude'] - lon) <= 0.3:
            return row['Country'], row['City']
    return None, None

# Create empty lists to store data for each month
monthly_data = []
for mon in range(1, 13):
    sim_df = xr.open_dataset(
        sim_dir + '{}.{}.CEDS01-fixed-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon),
        engine='netcdf4')  # CEDS, c360, noLUO
    # sim_df = xr.open_dataset(sim_dir + '{}.{}.EDGARv61-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon), engine='netcdf4') # EDGAR, noLUO
    # sim_df = xr.open_dataset(sim_dir + '{}.{}.HTAPv3-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon), engine='netcdf4') # HTAP, noLUO
    # sim_df = xr.open_dataset(sim_dir + '{}.{}.CEDS01-fixed-vert.GEOSFP-CSwinds.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon), engine='netcdf4')  # CEDS, c720, noLUO
    # sim_df = xr.open_dataset(sim_dir + '{}.{}.CEDS01-fixed-vert.GEOSFP.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon), engine='netcdf4')  # CEDS, c360, LUO
    # sim_df = xr.open_dataset(sim_dir + '{}.{}.CEDS01-fixed-vert.{}.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, meteorology, year, mon), engine='netcdf4') # CEDS, c180, noLUO, GEOS-FP
    # sim_df = xr.open_dataset(sim_dir + '{}.{}.CEDS01-fixed-vert.{}.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, meteorology, year, mon), engine='netcdf4')  # CEDS, c180, noLUO, MERRA2
    # Extract nf, Ydim, Xdim, lon/lat, buffer, and BC from simulation data
    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')
    sim_df['OA'] = sim_df['POA'] + sim_df['SOA']
    print(np.array(sim_df[species]).shape)
    sim_conc = np.array(sim_df[species])[0, :, :, :]  # Selecting the first level
    # sim_conc = np.array(sim_df[species]).reshape([6, 360, 360])
    # pw_conc = (pop * sim_conc) / np.nansum(pop)  # compute pw conc for each grid point, would be super small and not-meaningful

    # Load the Data
    obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/FT-IR_OM_OC_vs_Residual_Chris_vs_sim_OMOC.xlsx', sheet_name='All')
    # obs_df['OA'] = obs_df['RM_dry']
    # obs_df['OA'] = obs_df['FTIR_OM']
    obs_df['OA'] = obs_df.apply(lambda row: row['FTIR_OM'] if row['FTIR_OM']/row['FTIR_OC'] < 2.5 else row['FTIR_OC'] * 2.5, axis=1)
    obs_df['Longitude'] = obs_df['site_lon']
    obs_df['Latitude'] = obs_df['site_lat']
    # obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/Summary_measurements_2019.xlsx')
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])
    # Filter obs_df based on 'start_month'
    obs_df = obs_df[obs_df['start_month'] == mon]
    # Drop NaN and infinite values from obs_conc
    obs_df = obs_df.replace([np.inf, -np.inf], np.nan)  # Convert infinite values to NaN
    obs_df = obs_df.dropna(subset=[species], thresh=1)
    # Extract lon/lat and BC from observation data
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
    # Calculate distance between the observation and all simulation points
    for k in range(len(obs_lon)):
        # Spherical law of cosines:
        R = 6371  # Earth radius 6371 km
        buffer = 10  # 10-degree radius
        latk = obs_lat.iloc[k]  # Use .iloc to access value by integer location
        lonk = obs_lon.iloc[k]
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
    print(f'Measured_{species}_in_{mon} Mean: {mean_obs:.2f}, SD: {sd_obs:.2f}, Max: {max_obs:.2f}')

# Combine monthly data to create the annual DataFrame
monthly_df = pd.concat(monthly_data, ignore_index=True)
monthly_df['month'] = monthly_df['month'].astype(int)
# Calculate annual average and standard error for each site
annual_df = monthly_df.groupby(['country', 'city']).agg({
    'sim': ['mean', lambda x: np.std(x) / np.sqrt(len(x))],
    'obs': ['mean', lambda x: np.std(x) / np.sqrt(len(x))],
    'num_obs': 'sum',
    'lat': 'mean',
    'lon': 'mean'
}).reset_index()
annual_df.columns = ['country', 'city', 'sim', 'sim_se', 'obs', 'obs_se', 'num_obs', 'lat', 'lon']

# with pd.ExcelWriter(out_dir + '{}_{}_{}_vs_FTIR_OC_{}_{}.xlsx'.format(cres, inventory, deposition, species, year), engine='openpyxl') as writer:
#     monthly_df.to_excel(writer, sheet_name='Mon', index=False)
#     annual_df.to_excel(writer, sheet_name='Annual', index=False)

sim_df.close()
################################################################################################
# Create scatter plot for Residual vs GEOS-Chem OA, mean+se, colored by region
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
def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)
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
    # nmd = np.mean((sim - obs) / obs) * 100  # Percentage
    nmd = np.sum(sim - obs) / np.sum(obs) * 100
    # Calculate NRMSD
    rmsd = np.sqrt(np.mean((sim - obs) ** 2))
    mean_obs = np.mean(obs)
    nrmsd = (rmsd / mean_obs) * 100  # Percentage
    return {'NMD (%)': nmd, 'NRMSD (%)': nrmsd}
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = compr_df['obs_FTIR_OM']
compr_df['obs_se'] = compr_df['obs_se_FTIR_OM']
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


# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# Add 1:1 line with grey dash
plt.plot([-5, 70], [-5, 70], color='grey', linestyle='--', linewidth=1, zorder=1)
# Create scatter plot with white background, black border, and no grid
sns.set(style="whitegrid", font="Arial", font_scale=1.2)
# Create scatter plot
# Create scatter plot
scatterplot = sns.scatterplot(
    x='obs', y='sim', data=compr_df, hue='city',
    palette=city_palette, s=60, alpha=1, edgecolor='k',
    style='city', markers=city_marker, zorder=2
)

# Add error bars manually using plt.errorbar()
for _, row in compr_df.iterrows():
    plt.errorbar(
        row['obs'], row['sim'],
        xerr=row['obs_se'], yerr=row['sim_se'],
        fmt='none', ecolor='black', elinewidth=1, capsize=3, zorder=1
    )

border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid

# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)

# Create legend with custom order
# sorted_city_color_match = sorted(city_color_match, key=lambda x: (
#     list(region_mapping.keys()).index(get_region_for_city(x['city'])),
#     region_mapping[get_region_for_city(x['city'])].index(x['city'])
# ))
sorted_city_color_match = sorted(city_color_match, key=lambda x: compr_df.loc[compr_df['city'] == x['city'], 'obs'].values[0], reverse=False)

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
legend = plt.legend(handles=legend_handles, facecolor='white', markerscale=0.85, bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('FT-IR OM vs CEDS C360 noLUO OA', fontsize=16, fontname='Arial', y=1.03)
# plt.title('Imposing OM/OC = 2.5 Threshold', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([-2, 55])
plt.ylim([-2, 55])
plt.xticks([0, 10, 20, 30, 40, 50], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30, 40, 50], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['obs'], compr_df['sim'])
sns.regplot(x='obs', y='sim', data=compr_df,
            scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Perform the calculations for the entire dataset
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='obs', sim_col='sim')
nmd = nmd_nrmsd_results['NMD (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = compr_df['obs'].count()  # Count non-null values
plt.text(0.65, 0.05, f'y = {slope:.1f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}\n$N$ = {num_points}\nNMD = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=ax.transAxes, fontsize=16, color='black')

# Set labels
# plt.xlabel('FT-IR OC × GEOS-Chem OM/OC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.xlabel('FT-IR OM (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('GEOS-Chem OA (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_{}_{}_vs_FTIR_OM_{}_{:02d}.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()

################################################################################################
# Calculate dust in Abu Dhabi
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, 'AbuDhabi_RCFM_FT-IR_master.xlsx'), sheet_name='All')
abu_dhabi_df = compr_df[compr_df['City'] == 'Abu Dhabi'].copy()
# Define the columns that need to be converted (all _XRF_ng columns)
XRF_columns = [col for col in compr_df.columns if col.endswith('_XRF_ng')]
# Perform conversion for each of these columns
for col in XRF_columns:
    # Convert mass to concentration (ng/m3 to µg/m³)
    abu_dhabi_df[f'{col.replace("_XRF_ng", "")}'] = abu_dhabi_df[col] / abu_dhabi_df['Volume_m3'] / 1000  # Divide by 1000 to convert ng to µg

# ==== Dust Equation ====
# Calcule MAL based on K, Mg, Na for each filetr
MAL_default = 0.72
CF_default = 1.14
# Calculate MAL based on K, Mg, Na for each filter
abu_dhabi_df['MAL_corrected'] = (
    (1.20 * abu_dhabi_df['K'] / abu_dhabi_df['Al']) +
    (1.66 * abu_dhabi_df['Mg'] / abu_dhabi_df['Al']) +
    (1.35 * abu_dhabi_df['Na'] / abu_dhabi_df['Al'])
) / 1.89
abu_dhabi_df['Soil_default'] = (
    (1.89 * abu_dhabi_df['Al'] * (1 + MAL_default)) +
    (2.14 * abu_dhabi_df['Si']) +
    (1.40 * abu_dhabi_df['Ca']) +
    (1.36 * abu_dhabi_df['Fe']) +
    (1.67 * abu_dhabi_df['Ti'])
) * CF_default
abu_dhabi_df['Soil_corrected'] = (
    (1.89 * abu_dhabi_df['Al'] * (1 + abu_dhabi_df['MAL_corrected'])) +
    (2.14 * abu_dhabi_df['Si']) +
    (1.40 * abu_dhabi_df['Ca']) +
    (1.36 * abu_dhabi_df['Fe']) +
    (1.67 * abu_dhabi_df['Ti'])
) * CF_default

# # ==== Al & Si Attenuation Correction ====
# abu_dhabi_df['dust_loading'] = abu_dhabi_df['Soil_default'] * abu_dhabi_df['Volume_m3'] / 3.53 # Convert µg/m³ to µg and then to µg/cm²
# abu_dhabi_df['A'] = 0.78 - 8.6e-4 * abu_dhabi_df['dust_loading'] + 4.0e-7 *abu_dhabi_df['dust_loading'] ** 2 # A = 0.78 - 8.6e-4 * dust_loading + 4.0e-7 * dust_loading ** 2 #
#
# # Correct Al & Si values
# abu_dhabi_df['Si_corrected'] = abu_dhabi_df['Si'] / abu_dhabi_df['A']
# abu_dhabi_df['Al_corrected'] = abu_dhabi_df['Al'] * 0.77 / abu_dhabi_df['A']  # 0.77 adjustment for Al calibration

# Calculate and print statistics for MAL_corrected and Soil_corrected
for col in ['MAL_corrected', 'Soil_corrected', 'Soil_default']:
    mean_val = abu_dhabi_df[col].mean()
    median_val = abu_dhabi_df[col].median()
    min_val = abu_dhabi_df[col].min()
    max_val = abu_dhabi_df[col].max()
    se_val = abu_dhabi_df[col].std() / (len(abu_dhabi_df[col]) ** 0.5)
    print(f"Statistics for {col}:\nMean: {mean_val}\nMedian: {median_val}\nMin: {min_val}\nMax: {max_val}\nSE: {se_val}\n")
# Save results to a new Excel file
# output_file = os.path.join(out_dir, 'AbuDhabi_MAL.xlsx')
# abu_dhabi_df[['FilterID', 'MAL_corrected', 'Soil_corrected', 'Soil_default',
#               'Soil', 'Al', 'Si', 'Ca', 'Fe', 'Ti', 'K', 'Mg', 'Na']].to_excel(output_file, index=False)
################################################################################################
# Calculate and plot MAL
################################################################################################
# Read the file
AEAZ_df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/AEAZ_master.csv')
ILHA_df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/ILHA_master.csv')
ILNZ_df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/ILNZ_master.csv')
# Add the 'Site' column to each DataFrame
AEAZ_df['Site'] = 'Abu Dhabi'
ILHA_df['Site'] = 'Haifa'
ILNZ_df['Site'] = 'Rehovot'
# Concatenate the DataFrames into master_df
master_df = pd.concat([AEAZ_df, ILHA_df, ILNZ_df], ignore_index=True)
master_df[['Volume_m3', 'start_year']] = master_df[['Volume_m3', 'start_year']].apply(pd.to_numeric, errors='coerce')
master_df = master_df[master_df['start_year'].isin([2019, 2020, 2021, 2022, 2023, 2024])]
# Define the columns that need to be converted (all _XRF_ng columns)
XRF_columns = [col for col in master_df.columns if col.endswith('_XRF_ng')]
master_df[XRF_columns] = master_df[XRF_columns].apply(pd.to_numeric, errors='coerce')
# Perform conversion for each of these columns
for col in XRF_columns:
    # Convert mass to concentration (ng/m3 to µg/m³)
    master_df[f'{col.replace("_XRF_ng", "")}'] = master_df[col] / master_df['Volume_m3'] / 1000  # Divide by 1000 to convert ng to µg

# Drop NaN and negative values in the specified columns
columns_to_check = ['start_year', 'Volume_m3', 'Al', 'K', 'Mg', 'Na', 'Si', 'Ca', 'Fe', 'Ti']
master_df = master_df.dropna(subset=columns_to_check)  # Drop NaN values
master_df = master_df[(master_df[columns_to_check] > 0).all(axis=1)]  # Drop negative values
print(master_df.head())

# ==== Dust Equation ====
# Calcule MAL based on K, Mg, Na for each filetr
MAL_default = 0.72
CF_default = 1.14
# Calculate MAL based on K, Mg, Na for each filter
master_df['MAL_corrected'] = (
    (1.20 * master_df['K'] / master_df['Al']) +
    (1.66 * master_df['Mg'] / master_df['Al']) +
    (1.35 * master_df['Na'] / master_df['Al'])
) / 1.89
master_df['Soil_default'] = (
    (1.89 * master_df['Al'] * (1 + MAL_default)) +
    (2.14 * master_df['Si']) +
    (1.40 * master_df['Ca']) +
    (1.36 * master_df['Fe']) +
    (1.67 * master_df['Ti'])
) * CF_default
master_df['Soil_corrected'] = (
    (1.89 * master_df['Al'] * (1 + master_df['MAL_corrected'])) +
    (2.14 * master_df['Si']) +
    (1.40 * master_df['Ca']) +
    (1.36 * master_df['Fe']) +
    (1.67 * master_df['Ti'])
) * CF_default

# # ==== Al & Si Attenuation Correction ====
# abu_dhabi_df['dust_loading'] = abu_dhabi_df['Soil_default'] * abu_dhabi_df['Volume_m3'] / 3.53 # Convert µg/m³ to µg and then to µg/cm²
# abu_dhabi_df['A'] = 0.78 - 8.6e-4 * abu_dhabi_df['dust_loading'] + 4.0e-7 *abu_dhabi_df['dust_loading'] ** 2 # A = 0.78 - 8.6e-4 * dust_loading + 4.0e-7 * dust_loading ** 2 #
#
# # Correct Al & Si values
# abu_dhabi_df['Si_corrected'] = abu_dhabi_df['Si'] / abu_dhabi_df['A']
# abu_dhabi_df['Al_corrected'] = abu_dhabi_df['Al'] * 0.77 / abu_dhabi_df['A']  # 0.77 adjustment for Al calibration

# Calculate and print statistics for MAL_corrected and Soil_corrected
for col in ['MAL_corrected', 'Soil_corrected', 'Soil_default']:
    mean_val = master_df[col].mean()
    median_val = master_df[col].median()
    min_val = master_df[col].min()
    max_val = master_df[col].max()
    se_val = master_df[col].std() / (len(master_df[col]) ** 0.5)
    print(f"Statistics for {col}:\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}\nSE: {se_val:.2f}\n")
mean_MAL_corrected = (
    (1.20 * (master_df['K']).mean() / (master_df['Al']).mean()) +
    (1.66 * (master_df['Mg']).mean() / (master_df['Al']).mean()) +
    (1.35 * (master_df['Na']).mean() / (master_df['Al']).mean())
) / 1.89

print(f"mean_MAL_corrected: {mean_MAL_corrected:.4f}")


nmb = (master_df['Soil_corrected'] - master_df['Soil_default']).sum() / master_df['Soil_default'].sum()
print(f"NMB: {nmb:.4f}")

# # Save results to a new Excel file
# output_file = os.path.join(out_dir, 'AbuDhabi_MAL.xlsx')
# master_df[['FilterID', 'MAL_corrected', 'Soil_corrected', 'Soil_default', 'Soil', 'Al', 'Si', 'Ca', 'Fe', 'Ti', 'K', 'Mg', 'Na']].to_excel(output_file, index=False)

# Combine date
master_df['start_year'] = master_df['start_year'].astype(str).str.strip()
# Convert columns to numeric, filling invalid values with NaN, and then replace NaNs with 0 or a valid default
master_df['start_year'] = pd.to_numeric(master_df['start_year'], errors='coerce', downcast='integer')
master_df['start_year'] = master_df['start_year'].astype(int)
master_df['start_month'] = master_df['start_month'].astype(int)
master_df['start_day'] = master_df['start_day'].astype(int)
master_df['Date'] = pd.to_datetime(
    master_df['start_year'].astype(str) + '-' + master_df['start_month'].astype(str) + '-' + master_df[
        'start_day'].astype(str))

# Ensure the 'date' column is in datetime format
master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce')

# Drop rows where 'date' or 'MAL_corrected' is NaN
master_df = master_df.dropna(subset=['Date', 'MAL_corrected'])

# Create the plot
plt.figure(figsize=(10, 6))
plt.axhline(y=0.72, color='grey', linestyle='--', linewidth=1)
# Plot for each site with corresponding color
for site, color in zip(['Abu Dhabi', 'Haifa', 'Rehovot'], ['black', 'blue', 'red']):
    site_data = master_df[master_df['Site'] == site]  # Filter the data for the site
    plt.plot(site_data['Date'], site_data['MAL_corrected'], marker='o', color=color, linestyle='None', label=site)
# Add a legend
legend = plt.legend(facecolor='white', prop={'family': 'Arial','size': 12})
legend.get_frame().set_edgecolor('white')

# Add labels and title
plt.xlabel('Date', fontsize=16, family='Arial')
plt.ylabel('MAL', fontsize=16, family='Arial')
# plt.title('MAL: Abu Dhabi', fontsize=16, family='Arial')

# Adjust the font for the tick labels
plt.xticks(rotation=45, fontsize=16, family='Arial')
plt.yticks(fontsize=16, family='Arial')
plt.ylim([0, 15])
# plt.xticks([2019, 2020, 2021, 2022, 2023, 2024], fontname='Arial', size=16)
plt.yticks([0, 5, 10, 15], fontname='Arial', size=16)
# Remove grid lines if you want to get rid of them
plt.grid(False)

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'MAL_AbhDhabi_Haifa_Rehovot.svg', dpi=300)

plt.show()

################################################################################################
# Other Measurements: Map SPARTAN, others, and GCHP data for the entire year
################################################################################################
# Map SPARTAN and GCHP data for the entire year
plt.style.use('default')
plt.figure(figsize=(10, 5))
left = 0.03
bottom = 0.05
width = 0.94
height = 0.9
ax = plt.axes([left, bottom, width, height], projection=ccrs.Miller())
ax.coastlines(color=(0.4, 0.4, 0.4))
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor=(0.4, 0.4, 0.4))
ax.set_global()
ax.set_extent([-140, 180, -60, 63], crs=ccrs.PlateCarree()) # New Zealand extends beyond 160°E (reaching about 178°E)
# ax.set_extent([70, 130, 20, 50], crs=ccrs.PlateCarree()) # China
# ax.set_extent([-130, -60, 15, 50], crs=ccrs.PlateCarree()) # US
# ax.set_extent([-10, 30, 40, 60], crs=ccrs.PlateCarree()) # Europe
# ax.set_extent([-15, 25, 40, 60], crs=ccrs.PlateCarree()) # Europe with cbar
# ax.set_extent([-130, -60, 25, 60], crs=ccrs.PlateCarree()) # NA

# Define the colormap
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0), (0.7, 0, 0)] # white-blure-green-yellow-orange-red
# colors = [(0, 0, 0.6), (0, 0, 1),(0, 0.27, 0.8), (0.4, 0.5, 0.9), (0.431, 0.584, 1), (0.7, 0.76, 0.9), (0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)] # dark blue to light blue to light red to dark red
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)
vmax = 20

# Accumulate data for each face over the year
annual_conc = None
for face in range(6):
    print(f"Processing face {face}")
    for mon in range(1, 13):
        print("Opening file:", sim_dir + '{}.noLUO.CEDS01-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, year, mon))
        with xr.open_dataset(
            sim_dir + '{}.{}.CEDS01-fixed-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon), engine='netcdf4') as sim_df:  # CEDS
            x = sim_df.corner_lons.isel(nf=face)
            y = sim_df.corner_lats.isel(nf=face)
            sim_df['OA'] = sim_df['POA'] + sim_df['SOA']
            print(np.array(sim_df[species]).shape) # (72, 6, 360, 360)
            conc = sim_df[species].isel(lev=0, nf=face).load()
            if annual_conc is None:
                annual_conc = conc
            else:
                annual_conc = annual_conc + conc
        print("File closed.")
    # Calculate the annual average
    annual_conc /= 12
    # annual_conc = annual_conc.squeeze() # (1, 360, 360) to (360, 360)
    # annual_conc = annual_conc[0, :, :] # (72, 360, 360) to (360, 360)
    print(x.shape, y.shape, annual_conc.shape)
    # Plot the annual average data for each face
    im = ax.pcolormesh(x, y, annual_conc, cmap=cmap, transform=ccrs.PlateCarree(), vmin=0, vmax=vmax)

# Read annual comparison data
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_df['obs'] = compar_df['obs_FTIR_OM']
compar_df['sim'] = compar_df['sim']
compar_notna = compar_df[compar_df.notna().all(axis=1)]
# Adjust SPARTAN observations
# compar_notna.loc[compar_notna['source'] == 'SPARTAN', 'obs'] *= 1
lon, lat, obs, sim = compar_notna.lon, compar_notna.lat, compar_notna.obs, compar_notna.sim
print(compar_notna['source'].unique())
# Define marker sizes
s1 = [40] * len(obs)  # inner circle: Measurement
s2 = [120] * len(obs)  # outer ring: Simulation
markers = {'SPARTAN': 'o', 'other': 's'}
# Create scatter plot for other data points (squares)
for i, row in compar_notna.iterrows():
    source = row['source']
    if source != 'SPARTAN':  # Exclude SPARTAN data for now
        marker = markers.get(source, 'o')
        plt.scatter(x=row['lon'], y=row['lat'], c=row['obs'], s=s1[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=4)
        plt.scatter(x=row['lon'], y=row['lat'], c=row['sim'], s=s2[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=3)
# Create scatter plot for SPARTAN data points (circles)
for i, row in compar_notna.iterrows():
    source = row['source']
    if source == 'SPARTAN':  # Plot SPARTAN data
        marker = markers.get(source, 'o')
        plt.scatter(x=row['lon'], y=row['lat'], c=row['obs'], s=s1[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=4)
        plt.scatter(x=row['lon'], y=row['lat'], c=row['sim'], s=s2[i], marker=marker, edgecolor='black',
                    linewidth=0.5, vmin=0, vmax=vmax, transform=ccrs.PlateCarree(), cmap=cmap, zorder=3)
# Calculate mean and standard error for SPARTAN sites
spartan_data = compar_notna[compar_notna['source'] == 'SPARTAN']
# spartan_developed_data = spartan_data[spartan_data['marker'] != 'Global South']
# spartan_gs_data = spartan_data[spartan_data['marker'] == 'Global South']
other_data = compar_notna[compar_notna['source'] == 'other']
mean_obs = np.mean(spartan_data['obs'])
std_error_obs = np.std(spartan_data['obs']) / np.sqrt(len(spartan_data['obs']))
mean_sim = np.mean(spartan_data['sim'])
std_error_sim = np.std(spartan_data['sim']) / np.sqrt(len(spartan_data['sim']))
# Calculate NMD and NMB for SPARTAN sites
NMD_spartan = np.sum(np.abs(spartan_data['sim'] - spartan_data['obs'])) / np.sum(spartan_data['obs'])
NMB_spartan = np.sum(spartan_data['sim'] - spartan_data['obs']) / np.sum(spartan_data['obs'])
# Print the final values
print(f"Normalized Mean Difference at SPARTAN sites (NMD_spartan): {NMD_spartan:.4f}")
print(f"Normalized Mean Bias at SPARTAN sites (NMB_spartan): {NMB_spartan:.4f}")
# Add text annotations to the plot
ax.text(0.3, 0.04, f'NMD across SPARTAN sites = {NMD_spartan * 100:.0f}%', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.3, 0.14, f'Meas = {mean_obs:.1f} ± {std_error_obs:.2f} µg/m$^3$', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.3, 0.08, f'Sim at Meas = {mean_sim:.1f} ± {std_error_sim:.2f} µg/m$^3$', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.3, 0.02, f'Sim (Population-weighted) = {pwm:.1f} ± {pwse:.4f} µg/m$^3$', fontsize=14, fontname='Arial', transform=ax.transAxes)
# ax.text(0.92, 0.05, f'{year}', fontsize=14, fontname='Arial', transform=ax.transAxes)
# plt.title(f'BC Comparison: GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial') # PM$_{{2.5}}$

# Create an inset axes for the color bar at the left middle of the plot
cbar_axes = inset_axes(ax,
                           width='1.5%',
                           height='50%',
                           bbox_to_anchor=(-0.95, -0.45, 1, 1),  # (x, y, width, height) relative to top-right corner
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
cbar = plt.colorbar(im, cax=cbar_axes, orientation="vertical")
font_properties = font_manager.FontProperties(family='Arial', size=12)
cbar.set_ticks([0, 5, 10, 15, 20], fontproperties=font_properties)
cbar.ax.set_ylabel(f'{species} (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
cbar.ax.tick_params(axis='y', labelsize=12)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

# plt.savefig(out_dir + 'FigS2_WorldMap_{}_{}_{}_Sim_vs_SPARTAN_{}_{}_FTIR_OM.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()