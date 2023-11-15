import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

# Set the directory path
HIPS_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/Analysis-10May2023_Joshin/Results/'
Site_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
Out_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/UV-Vis_HIPS_Comparison/'
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Calculate the difference in BC fraction
merged_df['diff'] = merged_df['f_BC_UV-Vis'] - merged_df['f_BC_HIPS']

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]

# Drop rows where mass_ug is greater than 200
merged_df = merged_df.loc[merged_df['mass_ug'] <= 200]

# Rename to simplify coding
merged_df.rename(columns={"diff": "diff"}, inplace=True)
merged_df.rename(columns={"mass_ug": "mass"}, inplace=True)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(15, 6))

# Set x-axis to logarithmic scale
# ax.set_xscale('log')

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='mass', y='diff', data=merged_df, hue='City', s=25, alpha=1, ax=ax)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Modify legend background color and position
legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.45), loc='center left', fontsize=12)
# legend.get_frame().set_linewidth(0.0)  # remove legend border
legend.get_texts()[0].set_fontname("Arial")  # set fontname of the first label

# Set title, xlim, ylim, ticks, labels
plt.title('Difference in Black Carbon Fraction Measured by UV-Vis and HIPS v.s. PM Mass', fontname='Arial', fontsize=14, y=1.03)
plt.xlim([merged_df['mass'].min() - 5, merged_df['mass'].max() + 5])
plt.ylim([merged_df['diff'].min() - 0.05, merged_df['diff'].max() + 0.05])
plt.xticks(fontname='Arial', size=14)
plt.yticks(fontname='Arial', size=14)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add reference line at y = 0
# Add reference line at y = 0
ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.05, 0.80, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)
plt.xlabel('PM Mass (µg)', fontsize=14, color='black', fontname='Arial')
plt.ylabel('Δ Black Carbon Fraction (UV-Vis - HIPS)', fontsize=14, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
plt.savefig(os.path.join(Out_dir_path, "Diff_BC_Fraction_PM_Mass.tiff"), format="TIFF", dpi=300)
plt.show()