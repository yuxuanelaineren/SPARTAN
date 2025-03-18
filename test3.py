
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
species = 'OA'
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
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
otherMeas_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/'
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
