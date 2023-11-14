import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

################################################################################################
# Set the directory path
HIPS_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
UV_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/Analysis-10May2023_Joshin/Results/'
Out_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/UV-Vis_HIPS_Comparison/'

#################### create scatter plot for each site as subplot
# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC'] <= 1]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)

# Get a list of all unique site names
site_names = merged_df['Site'].unique()

# Count the number of non-empty plots
num_plots = 0
for site_name in site_names:
    site_data = merged_df.loc[merged_df['Site'] == site_name]
    if len(site_data) >= 2:
        num_plots += 1

# Adjust number of rows and columns based on number of non-empty plots
Num_Cols = 7
Num_Rows = int(np.ceil(num_plots / Num_Cols))
Fig_Size = (25, 10)

# Create a figure with subplots for each site
fig, axs = plt.subplots(Num_Rows, Num_Cols, figsize=Fig_Size, sharex=True, sharey=True)

# Create scatter plots for each site
sns.set(font='Arial')
plot_idx = 0
for site_name in site_names:
    # Get the data for the current site
    site_data = merged_df.loc[merged_df['Site'] == site_name]

    # Skip site if no data points
    if len(site_data) < 2:
        continue

    # Determine the row and column indices for the current plot
    row_idx = plot_idx // Num_Cols
    col_idx = plot_idx % Num_Cols

    # Create subplot for site
    ax = axs[row_idx, col_idx]
    plot_idx += 1

    # Create scatter plot
    scatterplot = sns.scatterplot(x='UV-Vis', y='HIPS', data=site_data, s=15, alpha=1, ax=ax)
    scatterplot.set_facecolor('white')
    border_width = 1
    for spine in scatterplot.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(border_width)
    scatterplot.grid(False)  # remove the grid

    # Show x and y ticks for the current subplot
    scatterplot.tick_params(axis='both', which='both', labelsize=10, length=4)
    scatterplot.xaxis.set_tick_params(labelbottom=True)
    scatterplot.yaxis.set_tick_params(labelleft=True)

    # Set axis limits for the current subplot
    ax.set_xlim([min(site_data[['UV-Vis', 'HIPS']].min().min(), 0) - 0.5,
                 max(site_data[['UV-Vis', 'HIPS']].max().max(), 0) + 0.5])
    ax.set_ylim([min(site_data[['UV-Vis', 'HIPS']].min().min(), 0) - 0.5,
                 max(site_data[['UV-Vis', 'HIPS']].max().max(), 0) + 0.5])

    # Add 1:1 line with grey dash
    ax.plot([min(site_data[['UV-Vis', 'HIPS']].min().min(), 0), max(site_data[['UV-Vis', 'HIPS']].max().max(), 0)],
            [min(site_data[['UV-Vis', 'HIPS']].min().min(), 0), max(site_data[['UV-Vis', 'HIPS']].max().max(), 0)],
            color='grey', linestyle='--', linewidth=1)

    # Add number of data points to the plot
    num_points = len(site_data)
    ax.text(0.05, 0.65, f'N = {num_points}', transform=ax.transAxes, fontsize=12)

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
        ax.text(0.05, 0.7, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
                 transform=plt.gca().transAxes, fontsize=12)

    # Set plot title as site name
    ax.set_title(site_name, fontname='Arial', fontsize=12, y=1.03)

# Loop through subplots and remove x and y labels
for ax in axs.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

# set x and y labels above the subplots
fig.text(0.5, 0.05, 'UV-Vis Black Carbon(µg/m$^3$)', ha='center', va='center', fontsize=14)
fig.text(0.05, 0.5, 'HIPS Black Carbon (µg/m$^3$)', ha='center', va='center', rotation='vertical', fontsize=14)

# Adjust vertical distance among subplots
fig.subplots_adjust(hspace=0.4)
fig.tight_layout()
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.1)

# plt.savefig("/Users/renyuxuan/Desktop/Research/RCFM/OM_RM_positive_each_site.tiff", format="TIFF", dpi=300)
plt.show()