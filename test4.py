
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
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

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
# Beijing: Plot seasonal variations
################################################################################################
# Read the data
df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/CHTS_master.csv')
BC_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3', 'BC_SSR_ug',
                'BC_HIPS_ug', 'Flags']
df = df[BC_columns].copy()
# Exclide invald data
df['Mass_type'] = pd.to_numeric(df['Mass_type'], errors='coerce')
df = df.loc[df['Mass_type'] == 1]
df[['start_year', 'start_month', 'start_day', 'Volume_m3', 'BC_SSR_ug']] = df[
                    ['start_year', 'start_month', 'start_day', 'Volume_m3', 'BC_HIPS_ug']].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['start_year', 'start_month', 'start_day', 'Volume_m3', 'BC_HIPS_ug'])
df = df[df['Volume_m3'] > 0]
df = df[df['BC_HIPS_ug'] > 0]
# Calculate BC concentration
df['BC_conc'] = df['BC_HIPS_ug'] / df['Volume_m3']

# Create a datetime column by combining year, month, and day
df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce').fillna(0).astype(int)
df['start_month'] = pd.to_numeric(df['start_month'], errors='coerce').fillna(0).astype(int)
df['start_day'] = pd.to_numeric(df['start_day'], errors='coerce').fillna(0).astype(int)
# Drop rows with invalid dates (e.g., month > 12 or day > 31)
df = df[(df['start_month'] >= 1) & (df['start_month'] <= 12) &
        (df['start_day'] >= 1) & (df['start_day'] <= 31)]
# Combine year, month, and day into a single datetime column
# df['datetime'] = pd.to_datetime(df[['start_year', 'start_month', 'start_day']])
df['datetime'] = pd.to_datetime(df['start_year'].astype(str) + '-' +
                                df['start_month'].astype(str).str.zfill(2) + '-' +
                                df['start_day'].astype(str).str.zfill(2))
# Sort data by datetime
df = df.sort_values('datetime')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['BC_conc'], marker='o', linestyle='None', markersize=8, markeredgewidth=0.5, markeredgecolor='black')
border_width = 1
plt.ylim([0, 8])

# Use MonthLocator or YearLocator for fewer ticks
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Alternatively, limit the number of ticks shown
# plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

plt.xticks(fontname='Arial', size=12, rotation=45)
plt.yticks([0, 2, 4, 6, 8], fontname='Arial', size=18)
plt.xlabel('Date', fontname='Arial', size=18)
plt.ylabel('HIPS BC Concentration (µg/m$^3$)', fontsize=18, fontname='Arial')
plt.title('Time Series of BC Concentration in Beijing', fontsize=20, color='black', fontname='Arial')
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/' + 'BC_TimeSeries_Beijing_HIPS.svg', dpi=300)
plt.show()