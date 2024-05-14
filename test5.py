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
print("sim_lon:", sim_lon)
sim_lat = np.array(sim_df.coords['lat']) # Length of sim_lat: 1800
print("sim_lat:", sim_lat)
sim_conc = np.array(sim_df['OMOC'])
# Interpolate sim_lat
# interp_func = interpolate.interp1d(np.arange(len(sim_lat)), sim_lat, kind='linear')
# sim_lat = interp_func(np.linspace(0, len(sim_lat) - 1, len(sim_lon)))
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

# Calculate distance between the observation and all simulation points using cdist
for k, (latk, lonk) in enumerate(zip(site_lat, site_lon)):
    # Calculate distances between the observation and all simulation points
    distances = cdist([[latk, lonk]], np.column_stack((sim_lat, sim_lon)), 'euclidean')[0]
    # Select simulation points within a buffer around the observation's lat/lon
    within_buffer_indices = np.where(distances <= buffer)[0]
    # print("Maximum index in within_buffer_indices:", np.max(within_buffer_indices))
    # print("Indices within buffer:", within_buffer_indices)
    # print("Value of sim_conc at index 2283:", sim_conc[0, within_buffer_indices[2283]])
    # print("Shape of sim_conc at index 2283:", sim_conc[0, within_buffer_indices[2283]].shape)
    if len(within_buffer_indices) > 0:
        # Extract relevant simulation data
        sim_conck = sim_conc[0, within_buffer_indices]
        sim_lonk = sim_lon[within_buffer_indices]
        sim_latk = sim_lat[within_buffer_indices]
        print("Shape of sim_conck:", sim_conck.shape)
        print("Shape of sim_lonk:", sim_lonk.shape)
        print("Shape of sim_latk:", sim_latk.shape)
        # Find the nearest simulation point within the buffer
        nearest_index = np.argmin(distances[within_buffer_indices])
        print("Nearest index:", nearest_index)
        nearest_sim_lon = sim_lonk[nearest_index]
        nearest_sim_lat = sim_latk[nearest_index]
        nearest_sim_conc = sim_conck[nearest_index]
        # Store matching data
        match_sim[k] = nearest_sim_conc[0]
        match_sim_lat[k] = nearest_sim_lat
        match_sim_lon[k] = nearest_sim_lon


# Get unique lat/lon and OM/OC at the same simulation box
coords_u = np.column_stack((match_sim_lat, match_sim_lon))
unique_indices = np.unique(coords_u, axis=0, return_index=True)[1]
match_lon_u = match_sim_lon[unique_indices]
match_lat_u = match_sim_lat[unique_indices]
match_sim_u = match_sim[unique_indices]

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