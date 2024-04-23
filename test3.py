import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

# Set the directory path
HIPS_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/'
################################################################################################
# Create scatter plot for all sites
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(out_dir, 'BC_HIPS_UV-Vis_SPARTAN.xlsx'))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)
merged_df.rename(columns={"City": "city"}, inplace=True)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='HIPS', y='UV-Vis', data=merged_df, s=50, alpha=1, ax=ax, edgecolor='k')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=10)
# legend.get_frame().set_linewidth(0.0)

# Set title, xlim, ylim, ticks, labels
plt.xlim([merged_df['HIPS'].min()-0.5, 20])
plt.ylim([merged_df['HIPS'].min()-0.5, 20])
plt.xticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
plt.yticks([0, 5, 10, 15, 20], fontname='Arial', size=18)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = merged_df['HIPS']
y = merged_df['HIPS']
plt.plot([merged_df['HIPS'].min(), merged_df['HIPS'].max()], [merged_df['HIPS'].min(), merged_df['HIPS'].max()], color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.1, 0.7, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=22)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['HIPS']) & ~np.isnan(merged_df['UV-Vis'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['HIPS'][mask], merged_df['UV-Vis'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='HIPS', y='UV-Vis', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=22)

plt.xlabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('UV-Vis Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(os.path.join(Out_dir_path, "BC_Concentration_Comparison_HIPS_SSR.tiff"), format="TIFF", dpi=300)
plt.show()
