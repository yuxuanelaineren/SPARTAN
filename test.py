import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

def get_tick_interval(data_range):
    """
    Calculate a reasonable tick interval based on the data range.

    Parameters:
    - data_range: The range of the data.

    Returns:
    - tick_interval: The calculated tick interval.
    """
    # Define the conditions and corresponding intervals
    conditions = [0.5 * 6, 1 * 6, 2 * 6, 5 * 6]
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
Out_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/UV-Vis_HIPS_Comparison/'

# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC'] <= 1]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)

# Get a list of all unique site names
site_names = merged_df['City'].unique()

# Loop through each subplot and plot the data for the corresponding site
for site_name in site_names:
    # Get the data for the current site
    site_data = merged_df.loc[merged_df['City'] == site_name]

    # Skip site if no data points
    if len(site_data) < 2:
        continue

    # Create a new figure and axis for each site
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create scatter plot
    scatterplot = sns.scatterplot(x='UV-Vis', y='HIPS', data=site_data, s=15, alpha=1, ax=ax)
    scatterplot.set_facecolor('white')
    border_width = 1
    for spine in scatterplot.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(border_width)
    scatterplot.grid(False)  # remove the grid

    # Show x and y ticks for the current subplot
    scatterplot.tick_params(axis='both', which='both', labelsize=10, length=4, width=1)
    scatterplot.xaxis.set_tick_params(labelbottom=True)
    scatterplot.yaxis.set_tick_params(labelleft=True)

    # Set axis limits for the current subplot
    x_min, x_max = site_data[['UV-Vis', 'HIPS']].min().min() - 0.2, site_data[['UV-Vis', 'HIPS']].max().max() + 0.5
    y_min, y_max = site_data[['UV-Vis', 'HIPS']].min().min() - 0.2, site_data[['UV-Vis', 'HIPS']].max().max() + 0.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Calculate reasonable tick intervals based on the range of data
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_ticks_interval = get_tick_interval(x_range)
    y_ticks_interval = get_tick_interval(y_range)

    # Set x and y ticks with calculated interval
    scatterplot.xaxis.set_major_locator(MultipleLocator(x_ticks_interval))
    scatterplot.yaxis.set_major_locator(MultipleLocator(y_ticks_interval))

    # Set tick labels font to Arial
    scatterplot.set_xticks(scatterplot.get_xticks())  # Set x tick locations explicitly
    scatterplot.set_xticklabels(scatterplot.get_xticks(), fontdict={'fontname': 'Arial', 'fontsize': 10})
    scatterplot.set_yticks(scatterplot.get_yticks())  # Set y tick locations explicitly
    scatterplot.set_yticklabels(scatterplot.get_yticks(), fontdict={'fontname': 'Arial', 'fontsize': 10})

    # Add 1:1 line with grey dash
    ax.plot([x_min, x_max], [y_min, y_max], color='grey', linestyle='--', linewidth=1)

    # Add the number of data points to the plot
    num_points = len(site_data)
    ax.text(0.05, 0.82, f'N = {num_points}', transform=ax.transAxes, fontsize=12)

    # Perform linear regression with NaN handling
    mask = ~np.isnan(site_data['UV-Vis']) & ~np.isnan(site_data['HIPS'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(site_data['UV-Vis'][mask], site_data['HIPS'][mask])

    # Check for NaN in results
    if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
        print("Linear regression results contain NaN values. Check the input data.")
    else:
        # Add linear regression line and text
        sns.regplot(x='UV-Vis', y='HIPS', data=site_data, scatter=False, ci=None,
                    line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1}, ax=ax)

        # Change the sign of the intercept for display
        intercept_display = abs(intercept)  # Use abs() to ensure a positive value
        intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

        # Update the text line with the adjusted intercept
        ax.text(0.05, 0.85, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
                transform=ax.transAxes, fontsize=12)

    # Set plot title as site name
    ax.set_title(site_name, fontname='Arial', fontsize=14)

    # Set x and y labels for the subplot
    ax.set_xlabel('UV-Vis Black Carbon (µg/m$^3$)', fontsize=14, fontname='Arial')
    ax.set_ylabel('HIPS Black Carbon (µg/m$^3$)', fontsize=14, fontname='Arial')

    # Adjust vertical distance among subplots
    fig.tight_layout()

    # Save each subplot as an individual image
    plt.savefig(os.path.join(Out_dir_path, "BC_{site_name}.tiff"), format="tiff", dpi=300)
    plt.show()
