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
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'
################################################################################################
# plot HIPS BC vs FTIR OC, color cell by no. of pairs
################################################################################################
# Read the file
merged_df = pd.read_excel(out_dir + 'BC_HIPS_EC_FTIR_SPARTAN.xlsx', sheet_name='All')

# Drop rows where f_BC is greater than 1
# merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]
# Rename to simplify coding
merged_df.rename(columns={"City": "city"}, inplace=True)
# Multiply 'obs' column by 0.6 to reduce its values
merged_df['BC'] *= 0.6
# Create a 2D histogram to divide the area into squares and count data points in each square
hist, xedges, yedges = np.histogram2d(merged_df['BC'], merged_df['EC'], bins=60)

# Determine the color for each square based on the number of pairs
colors = np.zeros_like(hist)
for i in range(len(hist)):
    for j in range(len(hist[i])):
        pairs = hist[i][j]
        colors[i][j] = pairs

# Define the custom color scheme gradient
colors = [(1, 1, 1), (0, 0.5, 1), (0, 1, 0), (1, 1, 0), (1, 0.5, 0), (1, 0, 0)]

# Create a custom colormap using the gradient defined
cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', colors)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the 2D histogram with the specified color scheme
sns.set(font='Arial')
scatterplot = plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, origin='lower')

# Display the original data points as a scatter plot
# plt.scatter(merged_df['HIPS'], merged_df['IBR'], color='black', s=10, alpha=0.5)

# Set title, xlim, ylim, ticks, labels
plt.xlim([-0.5, 13])
plt.ylim([-0.5, 13])
plt.xticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
plt.yticks([0, 3, 6, 9, 12], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
plt.plot([merged_df['BC'].min(), merged_df['EC'].max()], [merged_df['BC'].min(), merged_df['EC'].max()], color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=ax.transAxes, fontsize=18)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['BC']) & ~np.isnan(merged_df['EC'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['BC'][mask], merged_df['EC'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='BC', y='EC', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=18)

# Set labels
plt.xlabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('FTIR Elemental Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Create the colorbar and specify font properties
cbar_ax = fig.add_axes([0.68, 0.25, 0.02, 0.4])
cbar = plt.colorbar(label='Number of Pairs', cax=cbar_ax)
cbar.ax.set_ylabel('Number of Pairs', fontsize=14, fontname='Arial')
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 10, 20, 30, 40], fontname='Arial', fontsize=14)

# show the plot
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "HIPS_BC_vs_FTIR_EC_noPairs_Mac6.svg"), format="SVG", dpi=300)

plt.show()