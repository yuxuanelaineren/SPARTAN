import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
from pandas.tseries.offsets import DateOffset
################################################################################################
def get_tick_interval(data_range):
    """
    Calculate a reasonable tick interval based on the data range.

    Parameters:
    - data_range: The range of the data.

    Returns:
    - tick_interval: The calculated tick interval.
    """
    # Define the conditions and corresponding intervals
    conditions = [0.5 * 5, 1 * 5, 2 * 5, 5 * 5]
    intervals = [0.5, 1, 2, 5]

    # Find the appropriate interval based on the conditions
    for condition, interval in zip(conditions, intervals):
        if data_range < condition:
            tick_interval = interval
            break
    else:
        # If none of the conditions are met, default to the last interval
        tick_interval = intervals[-1]

    return tick_interval

# Set the directory path
HIPS_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/Analysis-10May2023_Joshin/Results/'
Site_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
Out_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/UV-Vis_HIPS_Comparison/'

################################################################################################
# Create scatter plot for all sites
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]
# merged_df = merged_df.loc[merged_df['BC_HIPS_(ug/m3)'] <= 10]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)
merged_df.rename(columns={"BC_SSR_(ug/m3)": "SSR"}, inplace=True)

# Drop rows with NaN values in the 'SSR' column
merged_df = merged_df.dropna(subset=['UV-Vis'])
merged_df = merged_df.dropna(subset=['HIPS'])

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='UV-Vis', y='HIPS', data=merged_df, hue='City', s=15, alpha=1, ax=ax)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Modify legend background color and position
legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.5), loc='center left', fontsize=12)
# legend.get_frame().set_linewidth(0.0)  # remove legend border
legend.get_texts()[0].set_fontname("Arial")  # set fontname of the first label

# Set title, xlim, ylim, ticks, labels
plt.title('Comparison of Black Carbon Concentration Measured by HIPS and UV-Vis', fontname='Arial', fontsize=14, y=1.03)
plt.xlim([merged_df['HIPS'].min()-0.5, 35])
plt.ylim([merged_df['HIPS'].min()-0.5, 35])
plt.xticks([0, 10, 20, 30], fontname='Arial', size=14)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=14)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = merged_df['HIPS']
y = merged_df['HIPS']
plt.plot([merged_df['HIPS'].min(), merged_df['HIPS'].max()], [merged_df['HIPS'].min(), merged_df['HIPS'].max()], color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.05, 0.81, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)

# Perform linear regression with NaN handling
mask = ~np.isnan(merged_df['UV-Vis']) & ~np.isnan(merged_df['HIPS'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['UV-Vis'][mask], merged_df['HIPS'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='UV-Vis', y='HIPS', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.05, 0.85, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=14)

plt.xlabel('UV-Vis Black Carbon Concentration (µg/m$^3$)', fontsize=14, color='black', fontname='Arial')
plt.ylabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=14, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
plt.savefig(os.path.join(Out_dir_path, "BC_Concentration_Comparison_HIPS_UV-Vis.tiff"), format="TIFF", dpi=300)
plt.show()