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
