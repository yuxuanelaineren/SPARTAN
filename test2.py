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
site_lat = site_df['Latitude']
print("site_lon:", site_lon)
print("site_lat:", site_lat)
# Extract lon/lat, and OMOC from sim
sim_lon = np.array(sim_df.coords['lon']) # Length of sim_lon: 3600
sim_lon[sim_lon > 180] -= 360
sim_lat = np.array(sim_df.coords['lat']) # Length of sim_lat: 1800
sim_conc = np.array(sim_df['OMOC'])
print("sim_lon:", sim_lon)
print("sim_lat:", sim_lat)
print("sim_conc:", sim_conc)
# # Interpolate sim_lat
# interp_func = interpolate.interp1d(np.arange(len(sim_lat)), sim_lat, kind='linear')
# sim_lat = interp_func(np.linspace(0, len(sim_lat) - 1, len(sim_lon)))
# # Interpolate sim_conc
# sim_conc = np.empty((1, len(sim_lat), len(sim_lon)))
# for i in range(len(sim_lon)):
#     f = interpolate.interp1d(sim_lat, sim_conc[0, :, i], kind='linear')
#     sim_conc[0, :, i] = f(sim_lat)
# print("sim_lon:", sim_lon)
# print("sim_lat:", sim_lat)
# print("sim_conc:", sim_conc)
print("Shape of sim_lon:", sim_lon.shape)
print("Shape of sim_lat:", sim_lat.shape)
print("Shape of sim_conc:", sim_conc.shape)
# Initialize lists to store data
sim_data = []
site_data = []

# Iterate over each site
for site_index, (site_lon, site_lat) in enumerate(zip(site_lon, site_lat)):
    print('site_data:', (site_lat, site_lon))
    # Calculate the distance between the site and all grid points in the simulation data
    # distances = cdist([(site_lat, site_lon)], np.column_stack((sim_lat, sim_lon))) # need same shape in sim_lon and sim_lat
    distances = cdist([(site_lat, site_lon)], list(zip(sim_lat.ravel(), sim_lon.ravel())))
    print("distances:", distances)
    print("Shape of distances:", distances.shape)
    # Find the indices of the minimum distance
    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    print("min_index:", min_index)
    # Extract the nearest simulation latitude, longitude, and concentration
    sim_lat_nearest = sim_lat[min_index[0]] # row index (latitude index)
    sim_lon_nearest = sim_lon[min_index[1]] # column index (longitude index)
    sim_conc_nearest = sim_conc[min_index[0], min_index[0], min_index[1]]

    # Append the data to the lists
    sim_data.append((sim_lat_nearest, sim_lon_nearest, sim_conc_nearest))
    site_data.append((site_lat, site_lon))
    print('site_data:', site_data)

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
