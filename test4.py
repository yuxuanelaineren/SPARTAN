import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

################################################################################################
# SPARTAN HIPS vs UV-Vis
################################################################################################
# Set the directory path
HIPS_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/'
IBR_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'
################################################################################################
# plot HIPS vs UV-Vis, color cell by no. of pairs
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(out_dir, 'BC_HIPS_IBR_SPARTAN.xlsx'))

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_IBR_(ug/m3)": "IBR"}, inplace=True)
merged_df.rename(columns={"City": "city"}, inplace=True)

# Create a 2D histogram to divide the area into squares and count data points in each square
hist, xedges, yedges = np.histogram2d(merged_df['HIPS'], merged_df['IBR'], bins=100)

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

# Create the colorbar and specify font properties
cbar = plt.colorbar(label='Number of Pairs')
cbar.ax.yaxis.set_tick_params(labelsize=14)
cbar.ax.set_ylabel('Number of Pairs', fontsize=16, fontname='Arial')
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(1)
cbar.set_ticks([0, 5, 10, 15, 20])


# Set title, xlim, ylim, ticks, labels
plt.xlim([merged_df['HIPS'].min()-0.5, 30])
plt.ylim([merged_df['HIPS'].min()-0.5, 30])
plt.xticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
plt.yticks([0, 4, 8, 12, 16], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = merged_df['HIPS']
y = merged_df['HIPS']
plt.plot([merged_df['HIPS'].min(), merged_df['HIPS'].max()], [merged_df['HIPS'].min(), merged_df['HIPS'].max()], color='grey', linestyle='--', linewidth=1)


# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=ax.transAxes, fontsize=18)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['HIPS']) & ~np.isnan(merged_df['IBR'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['HIPS'][mask], merged_df['IBR'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='HIPS', y='IBR', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=18)

plt.xlabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('IBR Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
# show the plot
plt.tight_layout()
# plt.savefig(os.path.join(out_dir, "BC_Comparison_HIPS_UV-Vis.tiff"), format="TIFF", dpi=300)
plt.show()
