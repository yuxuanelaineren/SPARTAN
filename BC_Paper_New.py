#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
from scipy.io import loadmat
import matplotlib.lines as mlines
from scipy.stats import linregress
cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-EDGARv61-vert-{}-output/monthly/'.format(cres.lower(), deposition) # EDGAR, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-HTAPv3-vert-{}-output/monthly/'.format(cres.lower(), deposition) # HTAP, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-CSwinds-output/monthly/'.format(cres.lower(), deposition) # CEDS, C3720, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, LUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, GEOS-FP
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-merra2-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, MERRA2
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
otherMeas_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/'
################################################################################################
# Extract BC_HIPS from masterfile and lon/lat from site.details
################################################################################################
# Function to read and preprocess data from master files
def read_master_files(obs_dir):
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350', 'MXMC-0006', 'NGIL-0309'
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
                # Calculate BC concentrations and fractions
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
# Summarize monthly and annual mean for SPARTAN BC
################################################################################################
# Writ summary statistics to Excel
obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='All')

# Group by 'Local Site Name', 'Month', 'Latitude', and 'Longitude'
obs_monthly_df = obs_df.groupby(['Site', 'Country',	'City', 'start_month']).agg(
    monthly_mean=('BC', 'mean'),
    monthly_median=('BC', 'median'),
    monthly_count=('BC', 'count')
).reset_index()

# Calculate the annual average 'ECf_Val' for each 'SiteName', 'Latitude', 'Longitude'
obs_annual_df = obs_monthly_df.groupby(['Site', 'Country',	'City']).agg(
    annual_mean=('monthly_mean', 'mean'),
    annual_median=('monthly_median', 'median'),
    annual_count=('monthly_count', 'sum'),
    annual_se=('monthly_mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # SE = std / sqrt(n)
).reset_index()

# Save the annual DataFrame as 'annual'
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', engine='openpyxl', mode='a') as writer:
    obs_monthly_df.to_excel(writer, sheet_name='Mon', index=False)
    obs_annual_df.to_excel(writer, sheet_name='Annual', index=False)
################################################################################################
# Effects of COVID-19 lockdown
################################################################################################
# Load the dataset
df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='All')

# Define affected (Jan 2020 to June 2021) and non-affected periods
affected_period = ((df['start_year'] == 2020) | ((df['start_year'] == 2021) & (df['start_month'] <= 6)))
non_affected_period = ((df['start_year'] == 2019) | ((df['start_year'] == 2021) & (df['start_month'] > 6)) | (df['start_year'] > 2021))
# affected_period = ((df['start_year'] == 2020) | (df['start_year'] == 2021))
# non_affected_period = ((df['start_year'] == 2019) | (df['start_year'] > 2021))

# Count filters by city for each period
affected_counts = df[affected_period].groupby('City').size().reset_index(name='Affected_Period_Count')
non_affected_counts = df[non_affected_period].groupby('City').size().reset_index(name='Non_Affected_Period_Count')
total_counts = df.groupby('City').size().reset_index(name='Total_Count')

# Merge results into a single dataframe
result = total_counts.merge(affected_counts, on='City', how='left')
result = result.merge(non_affected_counts, on='City', how='left')
result = result.fillna(0)
print(result)

# Merge the affected, non-affected, and total counts by 'City'
merged_counts = total_counts.merge(affected_counts, on='City', how='left') \
                            .merge(non_affected_counts, on='City', how='left') \
                            .fillna(0)
merged_counts['Non_Affected_Percentage'] = merged_counts['Non_Affected_Period_Count'] / merged_counts['Total_Count']
print(merged_counts[['City', 'Non_Affected_Percentage']])

# Optional: Save the merged counts to a CSV file
# merged_counts.to_csv("filter_counts_by_city.csv", index=False)
# Extract non-affected samples based on the 'non_affected_period'
non_affected_df = df[non_affected_period]
# Save the annual DataFrame as 'annual'
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', engine='openpyxl', mode='a') as writer:
    non_affected_df.to_excel(writer, sheet_name='not-affected-by-COVID_All', index=False)

# Group by 'Local Site Name', 'Month', 'Latitude', and 'Longitude'
obs_monthly_df = non_affected_df.groupby(['Site', 'Country',	'City', 'start_month']).agg(
    monthly_mean=('BC', 'mean'),
    monthly_median=('BC', 'median'),
    monthly_count=('BC', 'count')
).reset_index()

# Calculate the annual average 'ECf_Val' for each 'SiteName', 'Latitude', 'Longitude'
obs_annual_df = obs_monthly_df.groupby(['Site', 'Country',	'City']).agg(
    annual_mean=('monthly_mean', 'mean'),
    annual_median=('monthly_median', 'median'),
    annual_count=('monthly_count', 'sum'),
    annual_se=('monthly_mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # SE = std / sqrt(n)
).reset_index()

# Save the annual DataFrame as 'annual'
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', engine='openpyxl', mode='a') as writer:
    obs_monthly_df.to_excel(writer, sheet_name='non_affected_Mon', index=False)
    obs_annual_df.to_excel(writer, sheet_name='non_affected_Annual', index=False)
# Example dataset for simulated and measured concentrations (replace with actual data)
data = {
    "City": ["Abu Dhabi", "Melbourne", "Dhaka", "Bujumbura", "Halifax", "Sherbrooke", "Beijing", "Addis Ababa", "Bandung",
             "Haifa", "Rehovot", "Kanpur", "Seoul", "Ulsan", "Mexico City", "Ilorin", "Fajardo", "Kaohsiung", "Taipei",
             "Pasadena", "Johannesburg", "Pretoria"],
    "Csim": [2.603483532, 0.431163175, 4.747680126, 3.673715311, 0.224381786, 0.362798662, 1.385444595, 4.799646778,
             4.02492436, 0.845562015, 1.169340151, 3.833072212, 1.175011504, 0.7798648, 2.008797088, 2.326521987, 0.10290891,
             1.33695288, 0.830166517, 0.474454487, 2.381180572, 2.013390368], # affected by Covid
    "Cmeas": [2.673810294, 0.431163175, 5.56315254, 3.673715311, 0.23148047, 0.363877719, 1.398329746, 4.799646778, 3.663149692,
             0.845562015, 1.159011749, 3.833072212, 1.196440665, 0.7798648, 2.073496912, 2.982349549, 0.10684992, 1.33695288,
             0.830166517, 0.474454487, 2.381180572, 2.098747274] # full dataset
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the Normalized Mean Bias (NMB)
nmb = np.sum(df['Csim'] - df['Cmeas']) / np.sum(df['Cmeas'])

# Calculate the Normalized Mean Difference (NMD)
nmd = np.sum(np.abs(df['Csim'] - df['Cmeas'])) / np.sum(df['Cmeas'])

# Print the results
print(f"Normalized Mean Bias (NMB): {nmb}")
print(f"Normalized Mean Difference (NMD): {nmd}")
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
    # sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    print(np.array(sim_df[species]).shape)
    sim_conc = np.array(sim_df[species])[0, :, :, :]  # Selecting the first level
    # sim_conc = np.array(sim_df[species]).reshape([6, 360, 360])
    # pw_conc = (pop * sim_conc) / np.nansum(pop)  # compute pw conc for each grid point, would be super small and not-meaningful

    # Load the Data
    obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='All')
    # obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/Summary_measurements_2019.xlsx')
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'),
                            usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])
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

with pd.ExcelWriter(out_dir + '{}_{}_{}_vs_SPARTAN_{}_{}.xlsx'.format(cres, inventory, deposition, species, year), engine='openpyxl') as writer:
    monthly_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_df.to_excel(writer, sheet_name='Annual', index=False)

sim_df.close()
################################################################################################
# Create scatter plot: sim vs meas, color blue and red with two lines, Beijing grey out
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs']
compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 1.35*1] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g,
x_range_2 = [1.4*1, compr_df['obs'].max()]

# Define custom blue and red colors
blue_colors = [(0.7, 0.76, 0.9),  (0.431, 0.584, 1), (0.4, 0.5, 0.9), (0, 0.27, 0.8),  (0, 0, 1), (0, 0, 0.6)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
# Create custom colormap
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', blue_colors)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', red_colors)

# Create a custom color palette mapping each city to a color based on observed values
def map_city_to_color(city, obs):
    if city == 'Beijing':  # Mark Beijing grey
        return 'grey'
    elif x_range_1[0] <= obs <= x_range_1[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()) - 1)
        return blue_cmap(obs_index)
    elif x_range_2[0] <= obs <= x_range_2[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()) - 1)
        return red_cmap(obs_index)
    else:
        return 'black'

# city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]
city_palette = [map_city_to_color(city, obs) if city != 'Singapore' else blue_cmap(0.5)
                for city, obs in zip(compr_df['city'], compr_df['obs'])]
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
sns.set(font='Arial')
# Add 1:1 line with grey dash
plt.plot([-0.5, 6.2], [-0.5, 6.2], color='grey', linestyle='--', linewidth=1, zorder=1)
# # Add error bars
# for i in range(len(compr_df)):
#     ax.errorbar(compr_df['obs'].iloc[i], compr_df['sim'].iloc[i],
#                 xerr=compr_df['obs_se'].iloc[i], yerr=compr_df['sim_se'].iloc[i],
#                 fmt='none', color='k', alpha=1, capsize=2, elinewidth=1, zorder=1) # color=city_palette[i], color='k'
# Create scatter plot
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=2)

# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 11])
plt.ylim([-0.5, 11])
plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Perform linear regression for the first segment
mask_1 = (compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])
# mask_1 = ((compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])) | (compr_df['city'] == 'Singapore')
slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(compr_df['obs'][mask_1], compr_df['sim'][mask_1])
# Perform linear regression for the second segment
mask_2 = (compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])
# mask_2 = ((compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])) & (compr_df['city'] != 'Singapore')
slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(compr_df['obs'][mask_2], compr_df['sim'][mask_2])
# Plot regression lines
sns.regplot(x='obs', y='sim', data=compr_df[mask_1],
            scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display_1 = abs(intercept_1)
intercept_display_2 = abs(intercept_2)
intercept_sign_1 = '-' if intercept_1 < 0 else '+'
intercept_sign_2 = '-' if intercept_2 < 0 else '+'
plt.text(0.6, 0.83, f'y = {slope_1:.1f}x {intercept_sign_1} {intercept_display_1:.2f}\n$r^2$ = {r_value_1 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='blue')
plt.text(0.6, 0.61, f'y = {slope_2:.3f}x {intercept_sign_2} {intercept_display_2:.1f}\n$r^2$ = {r_value_2 ** 2:.5f}',
         transform=scatterplot.transAxes, fontsize=18, color='red')
# Add the number of data points for each segment
num_points_1 = mask_1.sum()
plt.text(0.6, 0.77, f'N = {num_points_1}', transform=scatterplot.transAxes, fontsize=18, color='blue')
num_points_2 = mask_2.sum()
plt.text(0.6, 0.55, f'N = {num_points_2}', transform=scatterplot.transAxes, fontsize=18, color='red')

# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig2_Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_MAC10_BeijingGrey.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()
################################################################################################
# Create scatter plot: sim vs meas, color blue and red with only one line, Beijing grey out
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs']
compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 1.35*1] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g,
x_range_2 = [1.4*1, compr_df['obs'].max()]

# Define custom blue and red colors
blue_colors = [(0.7, 0.76, 0.9),  (0.431, 0.584, 1), (0.4, 0.5, 0.9), (0, 0.27, 0.8),  (0, 0, 1), (0, 0, 0.6)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
# Create custom colormap
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', blue_colors)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', red_colors)

# Create a custom color palette mapping each city to a color based on observed values
def map_city_to_color(city, obs):
    if city == 'Beijing':  # Mark Beijing grey
        return 'grey'
    elif x_range_1[0] <= obs <= x_range_1[1]:
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

# # Print legend context (city name, color, marker)
# city_palette = {city: map_city_to_color(city, compr_df.loc[compr_df['city'] == city, 'obs'].iloc[0]) for city in sorted_cities}
# city_marker = {city: map_city_to_marker(city) for city in sorted_cities}
# print("Legend Context:")
# for city in sorted_cities:
#     print(f"City: {city}, Color: {city_palette[city]}, Marker: {city_marker[city]}")
# # Function to convert RGBA to HEX
# def rgba_to_hex(rgba):
#     return '#{:02x}{:02x}{:02x}'.format(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
# # Convert the context to the desired dictionary format
# city_legend = {
#     city: {'color': rgba_to_hex(city_palette[city]), 'marker': city_marker[city]}
#     for city in sorted_cities
# }
# # Print the resulting dictionary
# print("city_legend = {")
# for city, values in city_legend.items():
#     print(f"    '{city}': {{'color': '{values['color']}', 'marker': '{values['marker']}'}}{','}")
# print("}")

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
sns.set(font='Arial')
# Add 1:1 line with grey dash
plt.plot([-0.5, 5.6], [-0.5, 5.6], color='grey', linestyle='--', linewidth=1, zorder=1)
# # Add error bars
# for i in range(len(compr_df)):
#     ax.errorbar(compr_df['obs'].iloc[i], compr_df['sim'].iloc[i],
#                 xerr=compr_df['obs_se'].iloc[i], yerr=compr_df['sim_se'].iloc[i],
#                 fmt='none', color='k', alpha=1, capsize=2, elinewidth=1, zorder=1) # color=city_palette[i], color='k'
# Create scatter plot
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=2)

# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 9.5])
plt.ylim([-0.5, 9.5])
plt.xticks([0, 3, 6, 9], fontname='Arial', size=18)
plt.yticks([0, 3, 6, 9], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Define mask to count valid points
df_no_beijing = compr_df[compr_df['city'] != 'Beijing']
mask = df_no_beijing['obs'].notnull() & df_no_beijing['sim'].notnull()
num_points = mask.sum()
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df_no_beijing['obs'], df_no_beijing['sim'])
sns.regplot(x='obs', y='sim', data=df_no_beijing,
            scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
# Add text with regression equation and statistics
intercept_display_1 = abs(intercept)
intercept_sign_1 = '-' if intercept < 0 else '+'
plt.text(0.6, 0.73, f'y = {slope:.2f}x {intercept_sign_1} {intercept_display_1:.2f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}',
         transform=ax.transAxes, fontsize=18, color='k')
plt.text(0.85, 0.03, f'{inventory}', transform=scatterplot.transAxes, fontsize=18, color='k') # 0.8 for EDGAR and 0.3 for HTAP

# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig3a_Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_MAC10_BeijingGrey.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()
################################################################################################
# Create scatter plot: c360 vs c720, colored by major Sim vs Meas plot (blue and red)
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir + 'C720_CEDS_noLUO_202207_vs_C360_CEDS_noLUO_202207.xlsx'))
# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")
# Define city-to-color and marker mappings
city_legend = {
    'Fajardo': {'color': '#b2c2e6', 'marker': 'd'},
    'Halifax': {'color': '#91ace4', 'marker': 'd'},
    'Sherbrooke': {'color': '#6e95ff', 'marker': 'd'},
    'Melbourne': {'color': '#6a8af3', 'marker': '*'},
    'Pasadena': {'color': '#6680e6', 'marker': 'd'},
    'Ulsan': {'color': '#325fcf', 'marker': '^'},
    'Taipei': {'color': '#0044cc', 'marker': '^'},
    'Haifa': {'color': '#0021e6', 'marker': 'p'},
    'Rehovot': {'color': '#0000ff', 'marker': 'p'},
    'Seoul': {'color': '#0000cc', 'marker': '^'},
    'Kaohsiung': {'color': '#000099', 'marker': '^'},
    'Beijing': {'color': '#e68080', 'marker': '^'},
    'Mexico City': {'color': '#ef8686', 'marker': 'd'},
    'Pretoria': {'color': '#fa7070', 'marker': 'o'},
    'Johannesburg': {'color': '#ff5252', 'marker': 'o'},
    'Abu Dhabi': {'color': '#ff2929', 'marker': 'p'},
    'Ilorin': {'color': '#fd0000', 'marker': 'o'},
    'Bandung': {'color': '#eb0000', 'marker': 's'},
    'Bujumbura': {'color': '#d60000', 'marker': 'o'},
    'Kanpur': {'color': '#bc0000', 'marker': 's'},
    'Addis Ababa': {'color': '#9d0000', 'marker': 'o'},
    'Dhaka': {'color': '#800000', 'marker': 's'},
}

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Add 1:1 line with grey dash
plt.plot([-0.5, 22], [-0.5, 22], color='grey', linestyle='--', linewidth=1, zorder=1)

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(
    x='c360',
    y='c720',
    data=compr_df,
    hue='city',
    palette={city: city_legend[city]['color'] for city in city_legend},
    style='city',
    markers={city: city_legend[city]['marker'] for city in city_legend},
    s=80,
    alpha=1,
    edgecolor='k'
)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Customize legend to match the order and appearance
handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(city)] for city in city_legend if city in labels]
ordered_labels = [city for city in city_legend if city in labels]
legend = plt.legend(ordered_handles, ordered_labels, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 4.5])
plt.ylim([-0.5, 4.5])
plt.xticks([0, 1, 2, 3, 4], fontname='Arial', size=18)
plt.yticks([0, 1, 2, 3, 4], fontname='Arial', size=18)
# plt.xlim([-0.5, 22])
# plt.ylim([-0.5, 22])
# plt.xticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
# plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Define the range of x-values for the two segments
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
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
# plt.text(0.65, 0.03, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)
plt.text(0.75, 0.03, f'July, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('C360 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('C720 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS7_Scatter_C720_CEDS_202207_vs_C360_CEDS_202207_BlueRed.svg', dpi=300)
plt.show()
################################################################################################
# Create scatter plot: noLUO vs LUO, colored by major Sim vs Meas plot (blue and red)
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir + 'C360_CEDS_LUO_vs_C360_CEDS_noLUO_201907.xlsx'))
# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")
# Define city-to-color and marker mappings
city_legend = {
    'Fajardo': {'color': '#b2c2e6', 'marker': 'd'},
    'Halifax': {'color': '#91ace4', 'marker': 'd'},
    'Sherbrooke': {'color': '#6e95ff', 'marker': 'd'},
    'Melbourne': {'color': '#6a8af3', 'marker': '*'},
    'Pasadena': {'color': '#6680e6', 'marker': 'd'},
    'Ulsan': {'color': '#325fcf', 'marker': '^'},
    'Taipei': {'color': '#0044cc', 'marker': '^'},
    'Haifa': {'color': '#0021e6', 'marker': 'p'},
    'Rehovot': {'color': '#0000ff', 'marker': 'p'},
    'Seoul': {'color': '#0000cc', 'marker': '^'},
    'Kaohsiung': {'color': '#000099', 'marker': '^'},
    'Beijing': {'color': '#e68080', 'marker': '^'},
    'Mexico City': {'color': '#ef8686', 'marker': 'd'},
    'Pretoria': {'color': '#fa7070', 'marker': 'o'},
    'Johannesburg': {'color': '#ff5252', 'marker': 'o'},
    'Abu Dhabi': {'color': '#ff2929', 'marker': 'p'},
    'Ilorin': {'color': '#fd0000', 'marker': 'o'},
    'Bandung': {'color': '#eb0000', 'marker': 's'},
    'Bujumbura': {'color': '#d60000', 'marker': 'o'},
    'Kanpur': {'color': '#bc0000', 'marker': 's'},
    'Addis Ababa': {'color': '#9d0000', 'marker': 'o'},
    'Dhaka': {'color': '#800000', 'marker': 's'},
}

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Add 1:1 line with grey dash
plt.plot([-0.5, 22], [-0.5, 22], color='grey', linestyle='--', linewidth=1, zorder=1)

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(
    x='noLUO',
    y='LUO',
    data=compr_df,
    hue='city',
    palette={city: city_legend[city]['color'] for city in city_legend},
    style='city',
    markers={city: city_legend[city]['marker'] for city in city_legend},
    s=80,
    alpha=1,
    edgecolor='k'
)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Customize legend to match the order and appearance
handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(city)] for city in city_legend if city in labels]
ordered_labels = [city for city in city_legend if city in labels]
legend = plt.legend(ordered_handles, ordered_labels, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 7])
plt.ylim([-0.5, 7])
plt.xticks([0, 2, 4, 6], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6], fontname='Arial', size=18)
# plt.xlim([-0.5, 17])
# plt.ylim([-0.5, 17])
# plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
# plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Define the range of x-values for the two segments
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['noLUO'][mask], compr_df['LUO'][mask])
# Plot regression lines
sns.regplot(x='noLUO', y='LUO', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.1f}x {intercept_sign} {intercept_display:.3f}\n$r^2$ = {r_value ** 2:.1f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
# plt.text(0.65, 0.03, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)
plt.text(0.75, 0.03, f'July, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('BC with Default Scavenging (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('BC with Alternative Scavenging (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS4_Scatter_C360_CEDS_LUO_vs_C360_CEDS_noLUO_201907_BlueRed.svg', dpi=300)
plt.show()
################################################################################################
# Create scatter plot: GEOS-FP vs MERRA2, colored by major Sim vs Meas plot (blue and red)
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir + 'C180_CEDS_noLUO_MERRA2_vs_GEOS-FP_201907.xlsx'))
# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")
# Define city-to-color and marker mappings
city_legend = {
    'Fajardo': {'color': '#b2c2e6', 'marker': 'd'},
    'Halifax': {'color': '#91ace4', 'marker': 'd'},
    'Sherbrooke': {'color': '#6e95ff', 'marker': 'd'},
    'Melbourne': {'color': '#6a8af3', 'marker': '*'},
    'Pasadena': {'color': '#6680e6', 'marker': 'd'},
    'Ulsan': {'color': '#325fcf', 'marker': '^'},
    'Taipei': {'color': '#0044cc', 'marker': '^'},
    'Haifa': {'color': '#0021e6', 'marker': 'p'},
    'Rehovot': {'color': '#0000ff', 'marker': 'p'},
    'Seoul': {'color': '#0000cc', 'marker': '^'},
    'Kaohsiung': {'color': '#000099', 'marker': '^'},
    'Beijing': {'color': '#e68080', 'marker': '^'},
    'Mexico City': {'color': '#ef8686', 'marker': 'd'},
    'Pretoria': {'color': '#fa7070', 'marker': 'o'},
    'Johannesburg': {'color': '#ff5252', 'marker': 'o'},
    'Abu Dhabi': {'color': '#ff2929', 'marker': 'p'},
    'Ilorin': {'color': '#fd0000', 'marker': 'o'},
    'Bandung': {'color': '#eb0000', 'marker': 's'},
    'Bujumbura': {'color': '#d60000', 'marker': 'o'},
    'Kanpur': {'color': '#bc0000', 'marker': 's'},
    'Addis Ababa': {'color': '#9d0000', 'marker': 'o'},
    'Dhaka': {'color': '#800000', 'marker': 's'},
}

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Add 1:1 line with grey dash
plt.plot([-0.5, 22], [-0.5, 22], color='grey', linestyle='--', linewidth=1, zorder=1)

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(
    x='GEOS-FP',
    y='MERRA2',
    data=compr_df,
    hue='city',
    palette={city: city_legend[city]['color'] for city in city_legend},
    style='city',
    markers={city: city_legend[city]['marker'] for city in city_legend},
    s=80,
    alpha=1,
    edgecolor='k'
)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Customize legend to match the order and appearance
handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(city)] for city in city_legend if city in labels]
ordered_labels = [city for city in city_legend if city in labels]
legend = plt.legend(ordered_handles, ordered_labels, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 5])
plt.ylim([-0.5, 5])
plt.xticks([0, 1, 2, 3, 4, 5], fontname='Arial', size=18)
plt.yticks([0, 1, 2, 3, 4, 5], fontname='Arial', size=18)
# plt.xlim([-0.5, 12])
# plt.ylim([-0.5, 12])
# plt.xticks([0, 4, 8, 12], fontname='Arial', size=18)
# plt.yticks([0, 4, 8, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Define the range of x-values for the two segments
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['GEOS-FP'][mask], compr_df['MERRA2'][mask])
# Plot regression lines
sns.regplot(x='GEOS-FP', y='MERRA2', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
# plt.text(0.65, 0.03, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)
plt.text(0.75, 0.03, f'July, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('GEOS-FP Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('MERRA-2 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS3_C180_CEDS_noLUO_MERRA2_vs_GEOS-FP_201907_BlueRed.svg', dpi=300)
plt.show()
################################################################################################
# Create scatter plot: c360 vs c720, colored by region
################################################################################################
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')
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
compr_df = pd.read_excel(os.path.join(out_dir + 'C720_CEDS_noLUO_202207_vs_C360_CEDS_noLUO_202207.xlsx'))

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
x_range_1 = [compr_df['obs'].min(), 1.35]
x_range_2 = [1.35, compr_df['obs'].max()]
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
plt.xlim([-0.5, 4.5])
plt.ylim([-0.5, 4.5])
plt.xticks([0, 1, 2, 3, 4], fontname='Arial', size=18)
plt.yticks([0, 1, 2, 3, 4], fontname='Arial', size=18)
# plt.xlim([-0.5, 22])
# plt.ylim([-0.5, 22])
# plt.xticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
# plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['c360']
y = compr_df['c360']
plt.plot([compr_df['c360'].min(), 22], [compr_df['c360'].min(), 22],
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
# plt.text(0.78, 0.03, f'January', transform=scatterplot.transAxes, fontsize=18)
plt.text(0.88, 0.03, f'July', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('C360 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('C720 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS1_Scatter_C720_CEDS_202207_vs_C360_CEDS_202207.svg', dpi=300)
plt.show()
################################################################################################
# Create scatter plot: noLUO vs LUO, colored by region
################################################################################################
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')
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
compr_df = pd.read_excel(os.path.join(out_dir + 'C360_CEDS_LUO_vs_C360_CEDS_noLUO_201901.xlsx'))

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
x_range_1 = [compr_df['obs'].min(), 1.35]
x_range_2 = [1.35, compr_df['obs'].max()]
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='noLUO', y='LUO', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city',  markers=city_marker)
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
# plt.xlim([-0.5, 8])
# plt.ylim([-0.5, 8])
# plt.xticks([0, 2, 4, 6, 8], fontname='Arial', size=18)
# plt.yticks([0, 2, 4, 6, 8], fontname='Arial', size=18)
plt.xlim([-0.5, 17])
plt.ylim([-0.5, 17])
plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['noLUO']
y = compr_df['LUO']
plt.plot([-0.5, 17], [-0.5, 17], color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['noLUO'][mask], compr_df['LUO'][mask])
# Plot regression lines
sns.regplot(x='noLUO', y='LUO', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
plt.text(0.65, 0.03, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)
# plt.text(0.75, 0.03, f'July, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('BC with Default Scavenging (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('BC with Alternative Scavengingn (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigSX_Scatter_C360_CEDS_LUO_vs_C360_CEDS_noLUO_201901.svg', dpi=300)
plt.show()
################################################################################################
# Create scatter plot: GEOS-FP vs MERRA2, colored by region
################################################################################################
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')
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
compr_df = pd.read_excel(os.path.join(out_dir + 'C180_CEDS_noLUO_MERRA2_vs_GEOS-FP_201901.xlsx'))

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
x_range_1 = [compr_df['obs'].min(), 1.35]
x_range_2 = [1.35, compr_df['obs'].max()]
x_range = [compr_df['obs'].min(), compr_df['obs'].max()]
# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='GEOS-FP', y='MERRA2', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city',  markers=city_marker)
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
# plt.xlim([-0.5, 6])
# plt.ylim([-0.5, 6])
# plt.xticks([0, 2, 4, 6], fontname='Arial', size=18)
# plt.yticks([0, 2, 4, 6], fontname='Arial', size=18)
plt.xlim([-0.5, 12])
plt.ylim([-0.5, 12])
plt.xticks([0, 4, 8, 12], fontname='Arial', size=18)
plt.yticks([0, 4, 8, 12], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
x = compr_df['GEOS-FP']
y = compr_df['MERRA2']
plt.plot([-0.5, 12], [-0.5, 12], color='grey', linestyle='--', linewidth=1)

# Perform linear regression for all segments
mask = (compr_df['obs'] >= x_range[0]) & (compr_df['obs'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['GEOS-FP'][mask], compr_df['MERRA2'][mask])
# Plot regression lines
sns.regplot(x='GEOS-FP', y='MERRA2', data=compr_df[mask],
            scatter=False, ci=None, line_kws={'color': 'black', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
plt.text(0.05, 0.66, f'y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='black')

# Add the number of data points for each segment
num_points = mask.sum()
plt.text(0.05, 0.6, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=18, color='black')
plt.text(0.65, 0.03, f'January, {year}', transform=scatterplot.transAxes, fontsize=18)
# plt.text(0.75, 0.03, f'July, {year}', transform=scatterplot.transAxes, fontsize=18)

# Set labels
plt.xlabel('GEOS-FP Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('MERRA2 Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigS4_C180_CEDS_noLUO_MERRA2_vs_GEOS-FP_201901.svg', dpi=300)
plt.show()

################################################################################################
# Create scatter plot: sim vs meas, color by GDP per Captia
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs']
compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the country groups
red_group = ["Burundi", "Ethiopia", "India", "Bangladesh", "Nigeria", "Indonesia", "South Africa", "China", "Mexico"]
blue_group = ['Taiwan', "Korea", "PuertoRico", "Israel", "UAE", "Canada", "Australia", "United States", "Singapore"]
# Add a new column to classify each country as 'red' or 'blue'
compr_df['color'] = compr_df['country'].apply(lambda x: 'red' if x in red_group else ('blue' if x in blue_group else 'other'))
# Define a custom function to map each city to a color based on its country
def map_city_to_color(city):
    if compr_df.loc[compr_df['city'] == city, 'color'].values[0] == 'red':
        return 'red'
    elif compr_df.loc[compr_df['city'] == city, 'color'].values[0] == 'blue':
        return 'blue'
    else:
        return 'grey'
city_palette = [map_city_to_color(city) for city in compr_df['city']]
# sorted_cities = sorted(compr_df['city'].unique(), key=lambda city: (compr_df.loc[compr_df['city'] == city, 'country'].iloc[0], compr_df.loc[compr_df['city'] == city, 'obs'].iloc[0]))
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
sns.set(font='Arial')
# Add 1:1 line with grey dash
plt.plot([-0.5, 6.2], [-0.5, 6.2], color='grey', linestyle='--', linewidth=1, zorder=1)
# # Add error bars
# for i in range(len(compr_df)):
#     ax.errorbar(compr_df['obs'].iloc[i], compr_df['sim'].iloc[i],
#                 xerr=compr_df['obs_se'].iloc[i], yerr=compr_df['sim_se'].iloc[i],
#                 fmt='none', color='k', alpha=1, capsize=2, elinewidth=1, zorder=1) # color=city_palette[i], color='k'
# Create scatter plot
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=2)
# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# # Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 11])
plt.ylim([-0.5, 11])
plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Perform regression for blue group
blue_mask = compr_df['color'] == 'blue'
slope_blue, intercept_blue, r_value_blue, _, _ = stats.linregress(compr_df.loc[blue_mask, 'obs'], compr_df.loc[blue_mask, 'sim'])
sns.regplot(x='obs', y='sim', data=compr_df[blue_mask], scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
plt.text(0.6, 0.83, f'y = {slope_blue:.2f}x + {intercept_blue:.2f}\n$r^2$ = {r_value_blue**2:.2f}', color='blue', transform=ax.transAxes, fontsize=18)

# Perform regression for red group
red_mask = compr_df['color'] == 'red'
slope_red, intercept_red, r_value_red, _, _ = stats.linregress(compr_df.loc[red_mask, 'obs'], compr_df.loc[red_mask, 'sim'])
# sns.regplot(x='obs', y='sim', data=compr_df[red_mask], scatter=False, ci=None, line_kws={'color': 'red', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
plt.text(0.6, 0.61, f'y = {slope_red:.2f}x + {intercept_red:.2f}\n$r^2$ = {r_value_red**2:.2f}', color='red', transform=ax.transAxes, fontsize=18)
# Add the number of data points
num_points_1 = blue_mask.sum()
plt.text(0.6, 0.77, f'N = {num_points_1}', transform=scatterplot.transAxes, fontsize=18, color='blue')
num_points_2 = red_mask.sum()
plt.text(0.6, 0.55, f'N = {num_points_2}', transform=scatterplot.transAxes, fontsize=18, color='red')


# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_GDPperCaptia.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()

################################################################################################
# Other Measurements: Summarize EPA CSN data
################################################################################################
# Read the data
CSN_df = pd.read_csv(otherMeas_dir + 'CSN_daily_SPEC_2019_raw.csv')

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)
print(CSN_df.head(5))
print(CSN_df.columns)
unique_parameters = CSN_df['Parameter Name'].unique()
print(unique_parameters)

# Convert 'Date Local' to datetime format
CSN_df['Date Local'] = pd.to_datetime(CSN_df['Date Local'])
# Extract year and month to group by month
CSN_df['Month'] = CSN_df['Date Local'].dt.to_period('M')
# Filter for rows where 'Parameter Name' is 'EC PM2.5 LC TOR'
EC_df = CSN_df[CSN_df['Parameter Name'] == 'EC PM2.5 LC TOR']
# Group by 'Local Site Name', 'Month', 'Latitude', and 'Longitude'
EC_monthly_df = EC_df.groupby(['Local Site Name', 'Month', 'Latitude', 'Longitude', 'Units of Measure']).agg(
    monthly_mean=('Arithmetic Mean', 'mean'),
    monthly_count=('Arithmetic Mean', 'count')
).reset_index()

# Calculate the annual average 'ECf_Val' for each 'SiteName', 'Latitude', 'Longitude'
EC_annual_df = EC_monthly_df.groupby(['Local Site Name', 'Latitude', 'Longitude', 'Units of Measure']).agg(
    annual_mean=('monthly_mean', 'mean'),
    annual_count=('monthly_count', 'sum')
).reset_index()

EC_test_df = EC_df.head(250)
# Save the annual DataFrame as 'annual'
with pd.ExcelWriter(os.path.join(otherMeas_dir + 'Summary_CSN_EC_2019.xlsx'), engine='openpyxl', mode='w') as writer:
    EC_test_df.to_excel(writer, sheet_name='test', index=False)
    EC_monthly_df.to_excel(writer, sheet_name='mon', index=False)
    EC_annual_df.to_excel(writer, sheet_name='annual', index=False)
################################################################################################
# Other Measurements: Summarize IMPROVE EC data
################################################################################################
# Read the data
IMPROVE_df = pd.read_excel(otherMeas_dir + 'IMPROVE_EC_2019_raw.xlsx', sheet_name='Data')

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
# Other Measurements: Summarize EMEP EC data
################################################################################################
## 1. Compile EC in raw EMEP '.nas' data
EMEP_dir = otherMeas_dir + '/EMEP_EC_2019_raw/'
# Function to convert the floating-point number to datetime
# Jan (1-31), Feb (32-59), Mar (60-90), Apr (91-120), May (121-151), June (152-181),
# July (182-212), Aug (213-243), Sept (244-273), Oct (274-304), Nov (305-334),Dec(335-365)
def convert_to_datetime(float_number):
    base_date = datetime(2019, 1, 1)
    integer_part = int(float_number)
    fractional_part = float_number - integer_part
    date_part = base_date + timedelta(days=integer_part - 1)
    total_seconds = fractional_part * 24 * 3600
    time_part = timedelta(seconds=total_seconds)
    final_datetime = date_part + time_part
    return final_datetime

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
                # Extract station code
                station_code = next((line.split(':')[1].strip() for line in lines if line.startswith('Station code:')), '')
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
with pd.ExcelWriter(os.path.join(other_obs_dir, 'EMEP_EC_2019_raw.xlsx'), engine='openpyxl', mode='w') as writer:
    for station_code, data in all_data:
        # Convert data into DataFrame
        df = pd.DataFrame(data)

        # Assign headers from the first row
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        # Print the header of the DataFrame
        print(f"Headers for station {station_code}: {df.columns.tolist()}")

        # Convert 'starttime' and 'endtime' columns
        if 'starttime' in df.columns and 'endtime' in df.columns:
            df['start_time'] = df['starttime'].apply(convert_to_datetime)
            df['end_time'] = df['endtime'].apply(convert_to_datetime)
        df.to_excel(writer, sheet_name=station_code, index=False, header=True)

## 2. Manually delete duplicate EC columns, select valid mesurements (000) and change all headter to starttime, endtime, EC, flag_EC, start_time, end_time to generate EMEP_EC_2019_processed.xlsx.

## 3. Calaulte monthly average EC for each site
other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
EMEP_EC = pd.ExcelFile(other_obs_dir + 'EMEP_EC_2019_processed.xlsx')
# Dictionary to store summary data
summary_data = {'Station Code': []}

# Loop through each sheet
for sheet_name in EMEP_EC.sheet_names:
    # Read data from the current sheet
    df = pd.read_excel(EMEP_EC, sheet_name)
    # Extract station code from sheet name
    station_code = sheet_name.replace('EMEP_EC_', '').replace('_processed', '')
    # Convert 'start_time' column to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    # Extract month from 'start_time'
    df['Month'] = df['start_time'].dt.month
    # Group data by month
    grouped = df.groupby('Month')
    # Calculate number of measurements, average, and standard error for each month
    summary = grouped['EC'].agg(['count', 'mean', 'sem']).reset_index()
    # Filter months with more than 10 rows
    # summary = summary[summary['count'] > 10]
    # Add station code to the summary data dictionary
    summary_data['Station Code'].append(station_code)
    # Merge summary data with the main summary dictionary
    for month in range(1, 13):
        month_data = summary[summary['Month'] == month]
        if not month_data.empty:
            summary_data[f'{month}_mean'] = summary_data.get(f'{month}_mean', []) + [month_data['mean'].iloc[0]]
            summary_data[f'{month}_sem'] = summary_data.get(f'{month}_sem', []) + [month_data['sem'].iloc[0]]
            summary_data[f'{month}_count'] = summary_data.get(f'{month}_count', []) + [month_data['count'].iloc[0]]
        else:
            summary_data[f'{month}_mean'] = summary_data.get(f'{month}_mean', []) + [0]
            summary_data[f'{month}_sem'] = summary_data.get(f'{month}_sem', []) + [0]
            summary_data[f'{month}_count'] = summary_data.get(f'{month}_count', []) + [0]

# Create DataFrame from the summary data dictionary
summary_df = pd.DataFrame(summary_data)
# Reorder columns to have 'Station Code' at the beginning
cols = summary_df.columns.tolist()
cols = ['Station Code'] + [col for col in cols if col != 'Station Code']
summary_df = summary_df[cols]

# Write summary DataFrame to 'Summary' sheet of the Excel file
with pd.ExcelWriter(EMEP_EC, mode='a', engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

## 4. Summarize site information from raw file
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
                # need to manually calculate EC mean and std

                # Append extracted information to EC_df list
                EC_df.append([station_code, station_name, latitude, longitude, component, unit, analytical_measurement_technique, filename])

# Convert the list of lists into a DataFrame
df = pd.DataFrame(EC_df, columns=['Station Code', 'Station Name', 'Latitude', 'Longitude', 'Component', 'Unit', 'Analytical_Measurement_Technique', 'Filename'])

# Save the DataFrame as an Excel file
with pd.ExcelWriter(os.path.join(other_obs_dir, 'EMEP_EC_Summary.xlsx'), engine='openpyxl', mode='a') as writer:
    df.to_excel(writer, index=False)

# Print the counts
print(f"Processed files: {processed_files}")

## 5. Manually calculate mean EC from monthly data.

################################################################################################
# Other Measurements: Summarize NAPS BC data
################################################################################################
## 1. Load NAPS BC from Aaron
# Load the .mat file
mat_data = loadmat('/Volumes/rvmartin/Active/Shared/aaron.vandonkelaar/NAPS-20230814/NAPS-PM25-SPECIATION-2019.mat')

# Extracting relevant variables
NAPSLAT = mat_data['NAPSLAT'].flatten()
NAPSLON = mat_data['NAPSLON'].flatten()
NAPSID = mat_data['NAPSID'].flatten()
NAPSTYPE = mat_data['NAPSTYPE'].flatten()
NAPSdates = mat_data['NAPSdates']
NAPSSPEC = mat_data['NAPSspec']
print("NAPSID shape:", NAPSID.shape)
print("NAPSSPEC shape:", NAPSSPEC.shape)

# Create DataFrame for basic site info
NAPS_data = {
    'Station ID': NAPSID,
    'Latitude': NAPSLAT,
    'Longitude': NAPSLON,
    'Type': NAPSTYPE
}
df_NAPS = pd.DataFrame(NAPS_data)
# df_NAPS.to_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/NAPS_BC_2019_raw.xlsx', index=False)

# Index for BC
BC_INDEX = 2

# Initialize lists to hold summary statistics
station_ids = []
mean_bc = []
std_error_bc = []
count_bc = []

# Compute statistics for each station
for i, station_id in enumerate(NAPSID):
    bc_data = NAPSSPEC[i, :, BC_INDEX]  # Extract BC data for this station
    # Remove NaN values and negative values
    bc_data = bc_data[~np.isnan(bc_data)]
    bc_data = bc_data[bc_data > 0]

    if len(bc_data) > 0:
        mean_value = np.mean(bc_data)
        std_error_value = np.std(bc_data, ddof=1) / np.sqrt(len(bc_data))
        count_value = len(bc_data)
        # Append results
        station_ids.append(station_id)
        mean_bc.append(mean_value)
        std_error_bc.append(std_error_value)
        count_bc.append(count_value)
    else:
        # Handle cases with no data
        station_ids.append(station_id)
        mean_bc.append(np.nan)
        std_error_bc.append(np.nan)
        count_bc.append(0)

# Create DataFrame for summary statistics
summary_data = {
    'Station ID': station_ids,
    'Mean': mean_bc,
    'Standard Error': std_error_bc,
    'Count': count_bc
}
df_summary = pd.DataFrame(summary_data)

# Merge df_NAPS and df_summary based on 'Station ID'
df_BC = pd.merge(df_NAPS, df_summary, on='Station ID')
df_BC.to_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/Summary_NAPS_BC_2019.xlsx', index=False)

## 2. Manually delete Land Use Type of 'I'
################################################################################################
# Other Measurements: Combine other measurements and GCHP dataset based on lat/lon
################################################################################################
## 1. Match other measurements and GCHP
# Create empty lists to store data for each month
monthly_data = []
# Loop through each month
for mon in range(1, 13):
    sim_df = xr.open_dataset(sim_dir + '{}.{}.CEDS01-fixed-vert.PM25.RH35.NOx.O3.{}{:02d}.MonMean.nc4'.format(cres, deposition, year, mon), engine='netcdf4')  # CEDS, c360, noLUO
    obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/Summary_measurements_2019.xlsx')

    nf = np.array(sim_df.nf)
    Ydim = np.array(sim_df.Ydim)
    Xdim = np.array(sim_df.Xdim)
    sim_lon = np.array(sim_df.lons).astype('float32')
    sim_lon[sim_lon > 180] -= 360
    sim_lat = np.array(sim_df.lats).astype('float32')
    # sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    print(np.array(sim_df[species]).shape)
    sim_conc = np.array(sim_df[species])[0, :, :, :]  # Selecting the first level

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

    # Calculate distance between the observation and all simulation points
    for k in range(len(obs_lon)):
        # Spherical law of cosines:
        R = 6371  # Earth radius 6371 km
        buffer = 10
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

    # Function to find matching rows and add 'Country' and 'City'
    def find_and_add_location(lat, lon):
        for index, row in obs_df.iterrows():
            if abs(row['Latitude'] - lat) <= 0.3 and abs(row['Longitude'] - lon) <= 0.3:
                return row['Country'], row['City']
        return None, None
    compr_df[['country', 'city']] = compr_df.apply(lambda row: find_and_add_location(row['lat'], row['lon']), axis=1,
                                                   result_type='expand')
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
monthly_df = pd.concat(monthly_data, ignore_index=True)
annual_df = monthly_df.groupby(['country', 'city']).agg({
    'sim': ['mean', lambda x: np.std(x) / np.sqrt(len(x))],
    'obs': ['mean', lambda x: np.std(x) / np.sqrt(len(x))],
    'num_obs': 'sum',
    'lat': 'mean',
    'lon': 'mean'
}).reset_index()
annual_df.columns = ['country', 'city', 'sim', 'sim_se', 'obs', 'obs_se', 'num_obs', 'lat', 'lon']
with pd.ExcelWriter(out_dir + '{}_{}_{}_vs_other_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year), engine='openpyxl') as writer:
    monthly_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_df.to_excel(writer, sheet_name='Annual', index=False)
sim_df.close()

## 2. Manually combine SPARTAN and other sheets as 'C360_CEDS_noLUO_vs_SPARTAN_other_BC_2019_Summary.xlsx' and add column 'source'
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
vmax = 4

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
compar_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_other_{}_{}_Summary_forRevision_2month.xlsx'.format(cres, inventory, deposition, species, year)),
                          sheet_name='Annual')
compar_notna = compar_df[compar_df.notna().all(axis=1)]
# Adjust SPARTAN observations
compar_notna.loc[compar_notna['source'] == 'SPARTAN', 'obs'] *= 1
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
cbar.set_ticks([0, 1, 2, 3, 4], fontproperties=font_properties)
cbar.ax.set_ylabel(f'{species} (µg/m$^3$)', labelpad=10, fontproperties=font_properties)
cbar.ax.tick_params(axis='y', labelsize=12)
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)

# plt.savefig(out_dir + 'FigS2_WorldMap_{}_{}_{}_Sim_vs_SPARTAN_other_{}_{}_forRevision_2month.tiff'.format(cres, inventory, deposition, species, year), dpi=600)
plt.show()

################################################################################################
# Regression for two datasets
################################################################################################
# Data for regression
sim = [0.376899732, 0.382601509, 0.383543015, 0.21596839, 0.741094199, 0.385473877, 0.328784, 0.361103099, 0.311364786]
obs = [0.408375207, 0.332918632, 0.433083415, 0.195503968, 0.793203662, 0.390808532, 0.391118408, 0.345428075, 0.221408084]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(sim, obs)

# Calculate R^2
r_squared = r_value**2

# Print results
print("Slope:", slope)
print("Intercept:", intercept)
print("R-squared:", r_squared)
################################################################################################
# NMD and HMB for two datasets
################################################################################################
# Data
data = {
    "country": [
        "Australia", "Bangladesh", "Burundi", "Canada", "Canada", "China",
        "Ethiopia", "India", "Indonesia", "Israel", "Israel", "Korea",
        "Korea", "Mexico", "Nigeria", "PuertoRico", "South Africa",
        "South Africa", "Taiwan", "Taiwan", "UAE", "United States"
    ],
    "city": [
        "Melbourne", "Dhaka", "Bujumbura", "Halifax", "Sherbrooke", "Beijing",
        "Addis Ababa", "Kanpur", "Bandung", "Haifa", "Rehovot", "Seoul",
        "Ulsan", "Mexico City", "Ilorin", "Fajardo", "Johannesburg",
        "Pretoria", "Kaohsiung", "Taipei", "Abu Dhabi", "Pasadena"
    ],
    "sim": [
        0.512084121, 1.381082952, 2.547637429, 0.119816458, 0.18294857,
        10.37717696, 1.506690711, 2.693194926, 3.824371427, 1.283611243,
        2.110184163, 2.529543539, 0.956247906, 1.183777294, 1.440186777,
        0.019938074, 2.582517068, 2.000795275, 2.171044756, 3.038679928,
        1.674753805, 0.749298334
    ],
    "obs": [
        0.431163175, 5.56315254, 3.673715311, 0.23148047, 0.363877719,
        1.398329746, 4.799646778, 3.833072212, 3.663149692, 0.845562015,
        1.159011749, 1.196440665, 0.7798648, 2.073496912, 2.982349549,
        0.10684992, 2.381180572, 2.098747274, 1.33695288, 0.830166517,
        2.673810294, 0.474454487
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# # Calculate NMB and NMD
# total_sim_minus_obs = (df["sim"] - df["obs"]).sum()
# total_abs_sim_minus_obs = (df["sim"] - df["obs"]).abs().sum()
# total_obs = df["obs"].sum()

# Select the 11 sites with the highest observed values
top_11_obs = df.nlargest(10, "obs")
# Calculate NMB and NMD for the filtered data
total_sim_minus_obs = (top_11_obs["sim"] - top_11_obs["obs"]).sum()
total_abs_sim_minus_obs = (top_11_obs["sim"] - top_11_obs["obs"]).abs().sum()
total_obs = top_11_obs["obs"].sum()

nmb_percentage = (total_sim_minus_obs / total_obs) * 100
nmd_percentage = (total_abs_sim_minus_obs / total_obs) * 100

# Print results
print(f"Normalized Mean Bias (NMB): {nmb_percentage:.2f}%")
print(f"Normalized Mean Difference (NMD): {nmd_percentage:.2f}%")


################################################################################################
# Combine UV-Vis BC, FT-IR EC, and GCHP dataset based on lat/lon
################################################################################################
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
    # sim_df['BC_PM25'] = sim_df['BC'] / sim_df['PM25']
    print(np.array(sim_df[species]).shape)
    sim_conc = np.array(sim_df[species])[0, :, :, :]  # Selecting the first level
    # sim_conc = np.array(sim_df[species]).reshape([6, 360, 360])
    # pw_conc = (pop * sim_conc) / np.nansum(pop)  # compute pw conc for each grid point, would be super small and not-meaningful

    # Load the Data
    obs_df = pd.read_excel(
        '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC_old/BC_UV-Vis_SPARTAN/BC_UV-Vis_SPARTAN_Joshin_20230510.xlsx',
        usecols=['Filter ID', 'Sampling Start Date', 'f_BC', 'Mass_BC(ug/m3)', 'Location ID'])
    site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'),
                            usecols=['Site_Code', 'Country', 'City', 'Latitude', 'Longitude'])

    # Drop the last two digits in the 'Filter ID' column: 'AEAZ-0113-1' to 'AEAZ-0113'
    obs_df['FilterID'] = obs_df['Filter ID'].str[:-2]
    excluded_filters = [
        'AEAZ-0078', 'AEAZ-0086', 'AEAZ-0089', 'AEAZ-0090', 'AEAZ-0093', 'AEAZ-0097',
        'AEAZ-0106', 'AEAZ-0114', 'AEAZ-0115', 'AEAZ-0116', 'AEAZ-0141', 'AEAZ-0142',
        'BDDU-0346', 'BDDU-0347', 'BDDU-0349', 'BDDU-0350', 'MXMC-0006', 'NGIL-0309'
    ]
    # UV-Vis
    # Exclude specific FilterID values
    obs_df = obs_df[~obs_df['FilterID'].isin(excluded_filters)]
    obs_df.rename(columns={'Location ID': 'Site'}, inplace=True)
    obs_df.rename(columns={'Mass_BC(ug/m3)': 'BC'}, inplace=True)
    # Drop rows where fraction of UV-Vis BC > 0.8
    obs_df = obs_df.loc[obs_df['f_BC'] < 0.8]
    # Drop NaN and infinite values from UV-Vis BC concentrations
    obs_df = obs_df.replace([np.inf, -np.inf], np.nan)  # Convert infinite values to NaN
    obs_df = obs_df.dropna(subset=[species], thresh=1)

    # Extract the start_year and start_month from 'Sampling Start Date' column with the format 'YYYY/M/D'
    obs_df['start_year'] = pd.to_datetime(obs_df['Sampling Start Date']).dt.year
    obs_df['start_month'] = pd.to_datetime(obs_df['Sampling Start Date']).dt.month
    obs_df = obs_df[obs_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]
    obs_df = obs_df[obs_df['start_month'] == mon]
    # Merge 'Country', 'City', 'Latitude', 'Longitude' into obs_df based on 'Site'
    obs_df = obs_df.merge(site_df, how='left', left_on='Site', right_on='Site_Code')

    # # FT-IR
    # # Exclude specific FilterID values
    # obs_df = obs_df[~obs_df['FilterID'].isin(excluded_filters)]
    # obs_df.rename(columns={'EC_FTIR_ug/m3_raw': 'BC'}, inplace=True)
    # # Drop NaN and infinite values from UV-Vis BC concentrations
    # obs_df = obs_df.replace([np.inf, -np.inf], np.nan)  # Convert infinite values to NaN
    # obs_df = obs_df.dropna(subset=[species], thresh=1)
    # obs_df = obs_df[obs_df['start_year'].isin([2019, 2020, 2021, 2022, 2023])]
    # obs_df = obs_df[obs_df['start_month'] == mon]

    # Extract lon/lat and BC from obs
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

with pd.ExcelWriter(out_dir + '{}_{}_{}_vs_UV-Vis_{}_{}.xlsx'.format(cres, inventory, deposition, species, year),
                    engine='openpyxl') as writer:
    monthly_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_df.to_excel(writer, sheet_name='Annual', index=False)

sim_df.close()
################################################################################################
# Create scatter plot: sim vs meas, color blue and red with two lines
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_UV-Vis_{}_{}.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs']
compr_df['obs_se'] = 1 * compr_df['obs_se']
# Exclude the city 'Delhi' from compr_df
compr_df = compr_df[compr_df['city'] != 'Delhi']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 3*1] # 1.3 for FT-IR,
x_range_2 = [3*1, compr_df['obs'].max()]

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

# city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]
city_palette = [map_city_to_color(city, obs) if city != 'Singapore' else blue_cmap(0.5)
                for city, obs in zip(compr_df['city'], compr_df['obs'])]
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
sns.set(font='Arial')
# Add 1:1 line with grey dash
plt.plot([-0.5, 6.9], [-0.5, 6.9], color='grey', linestyle='--', linewidth=1, zorder=1)
# # Add error bars
# for i in range(len(compr_df)):
#     ax.errorbar(compr_df['obs'].iloc[i], compr_df['sim'].iloc[i],
#                 xerr=compr_df['obs_se'].iloc[i], yerr=compr_df['sim_se'].iloc[i],
#                 fmt='none', color='k', alpha=1, capsize=2, elinewidth=1, zorder=1) # color=city_palette[i], color='k'
# Create scatter plot
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=2)

# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 12])
plt.ylim([-0.5, 12])
plt.xticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
plt.yticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
# plt.xlim([-0.5, 11])
# plt.ylim([-0.5, 11])
# plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
# plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Perform linear regression for the first segment
mask_1 = (compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])
# mask_1 = ((compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])) | (compr_df['city'] == 'Singapore')
slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(compr_df['obs'][mask_1], compr_df['sim'][mask_1])
# Perform linear regression for the second segment
mask_2 = (compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])
# mask_2 = ((compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])) & (compr_df['city'] != 'Singapore')
slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(compr_df['obs'][mask_2], compr_df['sim'][mask_2])
# Plot regression lines
sns.regplot(x='obs', y='sim', data=compr_df[mask_1],
            scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)

# Add text with linear regression equations and other statistics
intercept_display_1 = abs(intercept_1)
intercept_display_2 = abs(intercept_2)
intercept_sign_1 = '-' if intercept_1 < 0 else '+'
intercept_sign_2 = '-' if intercept_2 < 0 else '+'
plt.text(0.6, 0.83, f'y = {slope_1:.2f}x {intercept_sign_1} {intercept_display_1:.2f}\n$r^2$ = {r_value_1 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='blue')
plt.text(0.6, 0.61, f'y = {slope_2:.2f}x {intercept_sign_2} {intercept_display_2:.2f}\n$r^2$ = {r_value_2 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='red')
# Add the number of data points for each segment
num_points_1 = mask_1.sum()
plt.text(0.6, 0.77, f'N = {num_points_1}', transform=scatterplot.transAxes, fontsize=18, color='blue')
num_points_2 = mask_2.sum()
plt.text(0.6, 0.55, f'N = {num_points_2}', transform=scatterplot.transAxes, fontsize=18, color='red')

# Set labels
plt.xlabel('UV-Vis Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'FigSX_Scatter_{}_{}_{}_vs_UV-Vis_{}_{:02d}.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()

################################################################################################
# Create scatter plot: sim vs meas, color blue and red with two lines, MAC 7 and 13 indicated as shades
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs'] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
compr_df['obs_high'] = (10 / 7) * compr_df['obs']
compr_df['obs_low'] = (10 / 13) * compr_df['obs']
compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 1.35*1] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
x_range_1_high = [compr_df['obs_high'].min(), 1.35*(10 / 7)] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
x_range_1_low = [compr_df['obs_low'].min(), 1.35*(10 / 13)] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
x_range_2 = [1.35*1, compr_df['obs'].max()]

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

# city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]
city_palette = [map_city_to_color(city, obs) if city != 'Singapore' else blue_cmap(0.5)
                for city, obs in zip(compr_df['city'], compr_df['obs'])]
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
sns.set(font='Arial')
# Add 1:1 line with grey dash
plt.plot([-0.5, 6.2], [-0.5, 6.2], color='grey', linestyle='--', linewidth=1, zorder=1)
# Add horizontal bars for uncertainty
for i, row in compr_df.iterrows():
    ax.plot([row['obs_low'], row['obs_high']], [row['sim'], row['sim']], color='black', alpha=0.7, linewidth=1.5, zorder=2)

# # Add error bars
# for i in range(len(compr_df)):
#     ax.errorbar(compr_df['obs'].iloc[i], compr_df['sim'].iloc[i],
#                 xerr=compr_df['obs_se'].iloc[i], yerr=compr_df['sim_se'].iloc[i],
#                 fmt='none', color='k', alpha=1, capsize=2, elinewidth=1, zorder=1) # color=city_palette[i], color='k'
# Create scatter plot
scatterplot = sns.scatterplot(x='obs', y='sim', data=compr_df, hue='city', palette=city_palette, s=80, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=2)

# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 11])
plt.ylim([-0.5, 11])
plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Perform linear regression for the first segment
mask_1 = (compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])
mask_1_high = (compr_df['obs_high'] >= x_range_1_high[0]) & (compr_df['obs_high'] <= x_range_1_high[1])
mask_1_low = (compr_df['obs_low'] >= x_range_1_low[0]) & (compr_df['obs_low'] <= x_range_1_low[1])
# mask_1 = ((compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])) | (compr_df['city'] == 'Singapore')
slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(compr_df['obs'][mask_1], compr_df['sim'][mask_1])
slope_1_high, intercept_1_high, r_value_1_high, p_value_1_high, std_err_1_high = stats.linregress(compr_df['obs_high'][mask_1_high], compr_df['sim'][mask_1_high])
slope_1_low, intercept_1_low, r_value_1_low, p_value_1_low, std_err_1_low = stats.linregress(compr_df['obs_low'][mask_1_low], compr_df['sim'][mask_1_low])
# Perform linear regression for the second segment
mask_2 = (compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])
# mask_2 = ((compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])) & (compr_df['city'] != 'Singapore')
slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(compr_df['obs'][mask_2], compr_df['sim'][mask_2])

# Plot regression lines
sns.regplot(x='obs', y='sim', data=compr_df[mask_1],
            scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
sns.regplot(x='obs_high', y='sim', data=compr_df[mask_1_high],
            scatter=False, ci=None, line_kws={'color': 'lightblue', 'linestyle': '--', 'linewidth': 0.5}, ax=ax)
sns.regplot(x='obs_low', y='sim', data=compr_df[mask_1_low],
            scatter=False, ci=None, line_kws={'color': 'lightblue', 'linestyle': '--', 'linewidth': 0.5}, ax=ax)

# Regression lines
y_range_1 = [slope_1 * x_range_1[0] + intercept_1, slope_1 * x_range_1[1] + intercept_1]
y_vals = np.linspace(y_range_1[0], y_range_1[1], 100)
regression_high_x = (y_vals - intercept_1_high) / slope_1_high
regression_low_x = (y_vals - intercept_1_low) / slope_1_low
# Fill the area between regression_low and regression_high with light blue
ax.fill_betweenx(y_vals, regression_low_x, regression_high_x, color='lightblue', alpha=0.5)

# Add text with linear regression equations and other statistics
intercept_display_1 = abs(intercept_1)
intercept_display_2 = abs(intercept_2)
intercept_sign_1 = '-' if intercept_1 < 0 else '+'
intercept_sign_2 = '-' if intercept_2 < 0 else '+'
plt.text(0.6, 0.83, f'y = {slope_1:.2f}x {intercept_sign_1} {intercept_display_1:.2f}\n$r^2$ = {r_value_1 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='blue')
plt.text(0.6, 0.61, f'y = {slope_2:.2f}x {intercept_sign_2} {intercept_display_2:.2f}\n$r^2$ = {r_value_2 ** 2:.2f}',
         transform=scatterplot.transAxes, fontsize=18, color='red')
# Add the number of data points for each segment
num_points_1 = mask_1.sum()
plt.text(0.6, 0.77, f'N = {num_points_1}', transform=scatterplot.transAxes, fontsize=18, color='blue')
num_points_2 = mask_2.sum()
plt.text(0.6, 0.55, f'N = {num_points_2}', transform=scatterplot.transAxes, fontsize=18, color='red')

# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig2_Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_MAC10+7+13.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()
################################################################################################
# Create scatter plot: sim vs meas, color blue and red with two lines, MAC 7 and 13 indicated as dots, Beijing grey out
################################################################################################
# Read the file
compr_df = pd.read_excel(os.path.join(out_dir, '{}_{}_{}_vs_SPARTAN_{}_{}_Summary.xlsx'.format(cres, inventory, deposition, species, year)), sheet_name='Annual')
compr_df['obs'] = 1 * compr_df['obs'] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
compr_df['obs_high'] = (10 / 7) * compr_df['obs']
compr_df['obs_low'] = (10 / 13) * compr_df['obs']
# compr_df['obs_se'] = 1 * compr_df['obs_se']

# Print the names of each city
unique_cities = compr_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Define the range of x-values for the two segments
x_range_1 = [compr_df['obs'].min(), 1.35*1] # 1 for MAC=10m2/g, 10/7 for MAC=7m2/g, 10/13 for MAC=13m2/g
x_range_1_high = [compr_df['obs_high'].min(), 1.35*(10 / 7)]
x_range_1_low = [compr_df['obs_low'].min(), 1.35*(10 / 13)]
x_range_2 = [1.4*1, compr_df['obs'].max()]
x_range_2_high = [1.4*(10 / 7), compr_df['obs_high'].max()]
x_range_2_low = [1.4*(10 / 13), compr_df['obs_low'].max()]

# Define custom blue and red colors
blue_colors = [(0.7, 0.76, 0.9),  (0.431, 0.584, 1), (0.4, 0.5, 0.9), (0, 0.27, 0.8),  (0, 0, 1), (0, 0, 0.6)]
red_colors = [(0.9, 0.6, 0.6), (1, 0.4, 0.4), (1, 0, 0), (0.8, 0, 0), (0.5, 0, 0)]
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', blue_colors)
red_cmap = LinearSegmentedColormap.from_list('red_cmap', red_colors)

# Map city to color based on observed values
def map_city_to_color(city, obs):
    if city == 'Beijing':  # Mark Beijing grey
        return 'grey'
    elif x_range_1[0] <= obs <= x_range_1[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_1[0], x_range_1[1])]['obs'].unique()) - 1)
        return blue_cmap(obs_index)
    elif x_range_2[0] <= obs <= x_range_2[1]:
        index_within_range = sorted(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()).index(obs)
        obs_index = index_within_range / (len(compr_df[compr_df['obs'].between(x_range_2[0], x_range_2[1])]['obs'].unique()) - 1)
        return red_cmap(obs_index)
    else:
        return 'black'

# city_palette = [map_city_to_color(city, obs) for city, obs in zip(compr_df['city'], compr_df['obs'])]
city_palette = [map_city_to_color(city, obs) if city != 'Singapore' else blue_cmap(0.5)
                for city, obs in zip(compr_df['city'], compr_df['obs'])]
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
sns.set(font='Arial')

# Add 1:1 line with grey dash
plt.plot([-0.5, 5.1], [-0.5, 5.1], color='grey', linestyle='--', linewidth=1, zorder=1)

# Plot horizontal uncertainty lines with markers at both ends
scatterplot = sns.scatterplot(x='obs_high', y='sim', data=compr_df, hue='city', palette=city_palette, s=50, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=3) # linestyle=(0, (3, 3))
sns.scatterplot(x='obs_low', y='sim', data=compr_df, hue='city', palette=city_palette, s=50, alpha=1, edgecolor='k', style='city', markers=city_marker, zorder=3)
for i, row in compr_df.iterrows():
    ax.plot([row['obs_low'], row['obs_high']], [row['sim'], row['sim']], color='black', alpha=0.7, linewidth=1.5, zorder=2)
# Overlay markers with vertical line style inside obs_high
for i, row in compr_df.iterrows():
    ax.plot(row['obs_low'], row['sim'], marker=r'$/$', markersize=3, color='black', alpha=0.7, linewidth=0, zorder=4)
    ax.plot(row['obs_high'], row['sim'], marker=r'$\backslash$', markersize=3, color='black', alpha=0.7, linewidth=0, zorder=4)


# Customize axis spines
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1)

# Customize legend markers
handles, labels = scatterplot.get_legend_handles_labels()
sorted_handles = [handles[list(labels).index(city)] for city in sorted_cities]
border_width = 1
# Customize legend order
legend = plt.legend(handles=sorted_handles, labels=sorted_cities, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=12, markerscale=1.25)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
# plt.title(f'GCHP-v13.4.1 {cres.lower()} {inventory} {deposition} vs SPARTAN', fontsize=16, fontname='Arial', y=1.03)  # PM$_{{2.5}}$
plt.xlim([-0.5, 11])
plt.ylim([-0.5, 11])
plt.xticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
plt.yticks([0, 2, 4, 6, 8, 10], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Perform linear regression for the first segment
# mask_1 = (compr_df['obs'] >= x_range_1[0]) & (compr_df['obs'] <= x_range_1[1])
mask_1_high = (compr_df['obs_high'] >= x_range_1_high[0]) & (compr_df['obs_high'] <= x_range_1_high[1])
mask_1_low = (compr_df['obs_low'] >= x_range_1_low[0]) & (compr_df['obs_low'] <= x_range_1_low[1])
# slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(compr_df['obs'][mask_1], compr_df['sim'][mask_1])
slope_1_high, intercept_1_high, r_value_1_high, p_value_1_high, std_err_1_high = stats.linregress(compr_df['obs_high'][mask_1_high], compr_df['sim'][mask_1_high])
slope_1_low, intercept_1_low, r_value_1_low, p_value_1_low, std_err_1_low = stats.linregress(compr_df['obs_low'][mask_1_low], compr_df['sim'][mask_1_low])
# Perform linear regression for the second segment
# mask_2 = (compr_df['obs'] >= x_range_2[0]) & (compr_df['obs'] <= x_range_2[1])
mask_2_high = (compr_df['obs_high'] >= x_range_2_high[0]) & (compr_df['obs_high'] <= x_range_2_high[1])
mask_2_low = (compr_df['obs_low'] >= x_range_2_low[0]) & (compr_df['obs_low'] <= x_range_2_low[1])
# slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(compr_df['obs'][mask_2], compr_df['sim'][mask_2])
slope_2_high, intercept_2_high, r_value_2_high, p_value_2_high, std_err_2_high = stats.linregress(compr_df['obs_high'][mask_2_high], compr_df['sim'][mask_2_high])
slope_2_low, intercept_2_low, r_value_2_low, p_value_2_low, std_err_2_low = stats.linregress(compr_df['obs_low'][mask_2_low], compr_df['sim'][mask_2_low])

# # Plot regression lines
# # sns.regplot(x='obs', y='sim', data=compr_df[mask_1],scatter=False, ci=None, line_kws={'color': 'blue', 'linestyle': '-', 'linewidth': 1.5}, ax=ax)
# sns.regplot(x='obs_high', y='sim', data=compr_df[mask_1_high],
#             scatter=False, ci=None, line_kws={'color': 'lightblue', 'linestyle': '--', 'linewidth': 0.5}, ax=ax)
# sns.regplot(x='obs_low', y='sim', data=compr_df[mask_1_low],
#             scatter=False, ci=None, line_kws={'color': 'lightblue', 'linestyle': '--', 'linewidth': 0.5}, ax=ax)
# # Regression lines
# y_range_1 = [slope_1 * x_range_1[0] + intercept_1, slope_1 * x_range_1[1] + intercept_1]
# y_vals = np.linspace(y_range_1[0], y_range_1[1], 100)
# regression_high_x = (y_vals - intercept_1_high) / slope_1_high
# regression_low_x = (y_vals - intercept_1_low) / slope_1_low
# # Fill the area between regression_low and regression_high with light blue
# ax.fill_betweenx(y_vals, regression_low_x, regression_high_x, color='lightblue', alpha=0.5)

# Add text with linear regression equations for high (right)
intercept_display_1_high = abs(intercept_1_high)
intercept_display_2_high = abs(intercept_2_high)
intercept_display_1_low = abs(intercept_1_low)
intercept_display_2_low = abs(intercept_2_low)
intercept_sign_1_high = '-' if intercept_1_high < 0 else '+'
intercept_sign_2_high = '-' if intercept_2_high < 0 else '+'
intercept_sign_1_low = '-' if intercept_1_high < 0 else '+'
intercept_sign_2_low = '-' if intercept_2_high < 0 else '+'
num_points_1 = mask_1_high.sum()
num_points_2 = mask_2_high.sum()
plt.text(0.5, 0.73, f'y$_L$ = {slope_1_low:.1f}x$_L$ {intercept_sign_1_low} {intercept_display_1_low:.2f}\ny$_H$ = {slope_1_high:.1f}x$_H$ {intercept_sign_1_high} {intercept_display_1_high:.2f}\n$r^2$ = {r_value_1_high ** 2:.2f}\nN = {num_points_1}',
         transform=scatterplot.transAxes, fontsize=18, color='blue')
plt.text(0.5, 0.48, f'y$_L$ = {slope_2_low:.3f}x$_L$ {intercept_sign_2_low} {intercept_display_2_low:.1f}\ny$_H$ = {slope_2_high:.4f}x$_H$ {intercept_sign_2_high} {intercept_display_2_high:.1f}\n$r^2$ = {r_value_2_high ** 2:.5f}\nN = {num_points_2}',
         transform=scatterplot.transAxes, fontsize=18, color='red')

# Set labels
plt.xlabel('HIPS Measured Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Simulated Black Carbon (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig(out_dir + 'Fig2_Scatter_{}_{}_{}_vs_SPARTAN_{}_{:02d}_MAC7+13_BeijingGrey.svg'.format(cres, inventory, deposition, species, year), dpi=300)
plt.show()

################################################################################################
# Create scatter plot for HIPS vs UV-Vis vs FT-IR, Dhaka
################################################################################################
# Read the file
compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC_old/BC_HIPS_UV-Vis_SPARTAN.xlsx', sheet_name='HIPS_UV-Uis')
compr_df['HIPS'] = compr_df['BC_HIPS_(ug/m3)']
compr_df['UV-Vis'] = compr_df['BC_UV-Vis_(ug/m3)']
# compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC_old/BC_HIPS_EC_FTIR_SPARTAN.xlsx', sheet_name='All')
# compr_df['HIPS'] = compr_df['BC']
# compr_df['FT-IR'] = compr_df['EC']
compr_df = compr_df[compr_df['City'].isin(['Dhaka'])]
# compr_df = compr_df[compr_df['City'].isin(['Dhaka, Addis Ababa'])]
# # Define custom color mapping
# color_map = {'Dhaka': 'red', 'Addis Ababa': 'blue'}
color_map = {'Dhaka': 'red'}

# Calculate mean and standard error grouped by City
summary_stats = compr_df.groupby('City').agg(
    OM_mean=('HIPS', 'mean'),
    OM_se=('HIPS', lambda x: x.std() / np.sqrt(len(x))),
    Residual_mean=('UV-Vis', 'mean'),
    Residual_se=('UV-Vis', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 7))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='HIPS', y='UV-Vis', data=compr_df, hue='City', palette=color_map, s=60, alpha=1, edgecolor='k', style='City')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Create legend with custom handles
legend = plt.legend(facecolor='white', bbox_to_anchor=(0.75, 0.05), loc='lower left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('HIPS vs UV-Vis', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [compr_df['HIPS'].min(), compr_df['HIPS'].max()]
# Perform linear regression for all segments
mask = (compr_df['HIPS'] >= x_range[0]) & (compr_df['HIPS'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['HIPS'][mask], compr_df['UV-Vis'][mask])
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
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='HIPS', sim_col='UV-Vis')
nmd = nmd_nrmsd_results['NMB (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.70, f'y = {slope:.1f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}\nNMB = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=16, color='black')

# Set labels
plt.xlabel('HIPS BC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('UV-Vis BC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_vs_UV-Vis_Dhaka.svg', dpi=300)

plt.show()
################################################################################################
# Match all FTIR_EC and HIPS_BC
################################################################################################
# Read data
FTIR_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/FTIR/'
HIPS_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='All')
HIPS_df['Date'] = pd.to_datetime(
    HIPS_df['start_year'].astype(str) + '-' + HIPS_df['start_month'].astype(str) + '-' + HIPS_df['start_day'].astype(str))
site_df = pd.read_excel(os.path.join(site_dir, 'Site_details.xlsx'), usecols=['Site_Code', 'Country', 'City'])
FTIR_b4_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch4_2024_03', usecols=['site', 'date', 'FTIR_EC'], skiprows=1)
FTIR_b2_b3_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch2_v3_batch3_v2', usecols=['site', 'date', 'FTIR_EC'], skiprows=1)
FTIR_b1_df = pd.read_excel(os.path.join(FTIR_dir, 'FTIR_summary_20250218.xlsx'), sheet_name='batch1_2020_08', usecols=['Site', 'Date', 'FTIR_EC'])
FTIR_b4_df.rename(columns={'FTIR_EC': 'EC', 'site': 'Site', 'date': 'Date'}, inplace=True)
FTIR_b2_b3_df.rename(columns={'FTIR_EC': 'EC', 'site': 'Site', 'date': 'Date'}, inplace=True)
# Add a 'batch' column to each DataFrame
FTIR_b4_df['batch'] = 'batch4_2024_03'
FTIR_b2_b3_df['batch'] = 'batch2_2022_06_batch3_2023_03'
FTIR_b1_df['batch'] = 'batch1_2020_08'
FTIR_df = pd.concat([FTIR_b4_df, FTIR_b2_b3_df]) # exlcude batch 1 as no lot specific calibrations
# Merge Residual and OM df based on matching values of "Site" and "Date"
merged_df = pd.merge(HIPS_df, FTIR_df, on=['Site', 'Date'], how='inner')
merged_df.rename(columns={'BC': 'BC'}, inplace=True)
# merged_df.rename(columns={'Country': 'country'}, inplace=True)
# merged_df.rename(columns={'City': 'city'}, inplace=True)
# Write to Excel
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_BC_FT-IR_EC_20250319.xlsx', engine='openpyxl', mode='w') as writer: # write mode
    merged_df.to_excel(writer, sheet_name='EC_batch234_HIPS_BC', index=False)

################################################################################################
# Create scatter plot for HIPS vs UV-Vis vs FT-IR, Dhaka
################################################################################################
# Read the file
compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC_old/BC_HIPS_UV-Vis_SPARTAN.xlsx', sheet_name='HIPS_UV-Uis')
compr_df['HIPS'] = compr_df['BC_HIPS_(ug/m3)']
compr_df['UV-Vis'] = compr_df['BC_UV-Vis_(ug/m3)']
# compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC_old/BC_HIPS_EC_FTIR_SPARTAN.xlsx', sheet_name='All')
# compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_BC_FT-IR_EC_20250319.xlsx', sheet_name='EC_batch234_HIPS_BC')
# compr_df['HIPS'] = compr_df['BC']
# compr_df['FT-IR'] = compr_df['EC']
compr_df = compr_df[compr_df['City'].isin(['Dhaka'])]
# compr_df = compr_df[compr_df['City'].isin(['Dhaka, Addis Ababa'])]
# # Define custom color mapping
# color_map = {'Dhaka': 'red', 'Addis Ababa': 'blue'}
color_map = {'Dhaka': 'red'}

# Calculate mean and standard error grouped by City
summary_stats = compr_df.groupby('City').agg(
    OM_mean=('HIPS', 'mean'),
    OM_se=('HIPS', lambda x: x.std() / np.sqrt(len(x))),
    Residual_mean=('UV-Vis', 'mean'),
    Residual_se=('UV-Vis', lambda x: x.std() / np.sqrt(len(x)))
).reset_index()

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 7))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='HIPS', y='UV-Vis', data=compr_df, hue='City', palette=color_map, s=60, alpha=1, edgecolor='k', style='City')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Create legend with custom handles
legend = plt.legend(facecolor='white', bbox_to_anchor=(0.75, 0.05), loc='lower left', fontsize=11.5)
legend.get_frame().set_edgecolor('black')

# Set title, xlim, ylim, ticks, labels
plt.title('HIPS vs UV-Vis', fontsize=18, fontname='Arial', y=1.03)
plt.xlim([0, 20])
plt.ylim([0, 20])
plt.xticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with grey dash
plt.plot([-10, 80], [-10, 80], color='grey', linestyle='--', linewidth=1)

# Define the range of x-values for the two segments
x_range = [compr_df['HIPS'].min(), compr_df['HIPS'].max()]
# Perform linear regression for all segments
mask = (compr_df['HIPS'] >= x_range[0]) & (compr_df['HIPS'] <= x_range[1])
slope, intercept, r_value, p_value, std_err = stats.linregress(compr_df['HIPS'][mask], compr_df['UV-Vis'][mask])
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
nmd_nrmsd_results = calculate_nmd_and_nrmsd(compr_df, obs_col='HIPS', sim_col='UV-Vis')
nmd = nmd_nrmsd_results['NMB (%)']
nrmsd = nmd_nrmsd_results['NRMSD (%)']

# Add text with linear regression equations and other statistics
intercept_display = abs(intercept)
intercept_sign = '-' if intercept < 0 else '+'
num_points = mask.sum()
plt.text(0.05, 0.70, f'y = {slope:.1f}x {intercept_sign} {intercept_display:.1f}\n$r^2$ = {r_value ** 2:.2f}\nN = {num_points}\nNMB = {nmd:.0f}%\nNRMSD = {nrmsd:.0f}%',
         transform=scatterplot.transAxes, fontsize=16, color='black')

# Set labels
plt.xlabel('HIPS BC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('UV-Vis BC (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Show the plot
plt.tight_layout()
# plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/HIPS_vs_UV-Vis_Dhaka.svg', dpi=300)

plt.show()