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
from scipy import interpolate

# Set the directory path
FTIR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/'
Residual_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Public_Data/RCFM/'
OMOC_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/FTIR_OC_OMOC_Residual/OM_OC/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/RCFM/'
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
site_lat = obs_df['Latitude']
print("site_lon:", site_lon)

# Extract lon/lat, and OMOC from sim
sim_lon = np.array(sim_df.coords['lon']) # Length of sim_lon: 3600
sim_lon[sim_lon > 180] -= 360
print("sim_lon_raw:", sim_lon)
sim_lat = np.array(sim_df.coords['lat']) # Length of sim_lat: 1800
print("sim_lat_raw:", sim_lat)
sim_conc = np.array(sim_df['OMOC'])
print("sim_OMOC_raw:", sim_conc)
# Interpolate sim_lat
# interp_func = interpolate.interp1d(np.arange(len(sim_lat)), sim_lat, kind='linear')
# sim_lat = interp_func(np.linspace(0, len(sim_lat) - 1, len(sim_lon)))
# print("sim_lat_interpolated:", sim_lat)
# Truncate sim_lon
# sim_lon = sim_lon[:len(sim_lat)]
print("sim_lon:", sim_lon)
print("Shape of sim_conc:", sim_conc.shape)
print("Shape of sim_lon:", sim_lon.shape)
print("Shape of sim_lat:", sim_lat.shape)

# Define buffer in degree
buffer = 10

# Initialize arrays to store matching data
match_obs = np.zeros(len(site_lon))
match_sim = np.full(len(site_lon), np.nan)
match_sim_lat = np.zeros(len(site_lon))
match_sim_lon = np.zeros(len(site_lon))

# Calculate distance between the observation and all simulation points
for index, (latk, lonk) in site_df[['Latitude', 'Longitude']].iterrows():
    # Spherical law of cosines:
    R = 6371  # Earth radius 6371 km
    # Select simulation points within a buffer around the observation's lat/lon
    ind = np.where((sim_lon > lonk - buffer) & (sim_lon < lonk + buffer)
                    & (sim_lat > latk - buffer) & (sim_lat < latk + buffer))
    print("Maximum index:", np.max(ind))
    print("Indices:", ind)
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
    match_sim[index] = np.nanmean(sim_conck[ii])
    match_sim_lat[index] = np.nanmean(sim_latk[ii])
    match_sim_lon[index] = np.nanmean(sim_lonk[ii])

# Get unique lat/lon and average observation data at the same simulation box
coords = np.concatenate((match_sim_lat[:, None], match_sim_lon[:, None]), axis=1)
coords_u, ind, ct = np.unique(coords, return_index=True, return_counts=True, axis=0)
match_lon_u = match_sim_lon[ind]
match_lat_u = match_sim_lat[ind]
match_sim_u = match_sim[ind]

columns = ['lat', 'lon', 'OMOC']
merged_df = np.concatenate((match_lat_u[:, None], match_lon_u[:, None], match_sim_u[:, None]), axis=1)
merged_df = pd.DataFrame(data=merged_df, index=None, columns=columns)

# Function to find matching rows and add 'Country' and 'City'
def find_and_add_location(lat, lon):
    for index, row in site_df.iterrows():
        if abs(row['Latitude'] - lat) <= 0.3 and abs(row['Longitude'] - lon) <= 0.3:
            return row['Country'], row['City']
    return None, None

# Apply the find_and_add_location function
merged_df[['country', 'city']] = merged_df.apply(lambda row: find_and_add_location(row['lat'], row['lon']), axis=1,
                                               result_type='expand')

# Print the resulting DataFrame
print(merged_df)

with pd.ExcelWriter(out_dir + 'OMOC_FTIROC_Residual_Summary.xlsx', engine='openpyxl') as writer:
    merged_df.to_excel(writer, sheet_name='DJF', index=False)
