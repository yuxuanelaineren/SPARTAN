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

# # Combine HIPS and UV-Vis dataset
################################################################################################
# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Create an empty dictionary to store the row count by site name
HIPS_count = {}
UV_count = {}

# Create an empty list to store individual HIPS DataFrames
HIPS_dfs = []

# # extract BC_HIPS from the master file folder
# Iterate over each file in the directory
for filename in os.listdir(HIPS_dir_path):
    if filename.endswith('.csv'):
        # Read the data from the master file
        master_data = pd.read_csv(os.path.join(HIPS_dir_path, filename), encoding='ISO-8859-1')

        # Print the columns to identify the available columns
        # print(f"Columns in {filename}: {master_data.columns}")

        # Specify the required columns
        HIPS_columns = ['FilterID', 'mass_ug', 'Volume_m3', 'BC_SSR_ug', 'BC_HIPS_ug']

        # Check if all required columns are present
        if all(col in master_data.columns for col in HIPS_columns):
            # Remove leading/trailing whitespaces from column names
            master_data.columns = master_data.columns.str.strip()
            # Select the specified columns
            HIPS_df = master_data[HIPS_columns].copy()

            # Extract the site name from the filename
            site_name = filename.split('_')[0]
            # Add the site name as a column in the selected data
            HIPS_df.loc[:, "Site"] = site_name  # safer to use .loc

            # Count the number of rows
            row_count = HIPS_df.shape[0]
            # Add the row count to the dictionaries
            if site_name in HIPS_count:
                HIPS_count[site_name] += row_count
            else:
                HIPS_count[site_name] = row_count

            # Append the current HIPS_df to the list
            HIPS_dfs.append(HIPS_df)
        else:
            print(f"Skipping {filename} because not all required columns are present.")

# Concatenate all HIPS DataFrames into a single DataFrame
HIPS_df = pd.concat(HIPS_dfs, ignore_index=True)

# # extract BC_UV-VIs from Joshin file folder
# Read the UV-Vis data from Analysis_ALL
UV_df = pd.read_excel(os.path.join(UV_dir_path, "Analysis_ALL.xlsx"), usecols=["Filter ID", "f_BC", "Location ID"])

# Drop the last two digits in the "Filter ID" column: 'AEAZ-0113-1' to 'AEAZ-0113'
UV_df["FilterID"] = UV_df["Filter ID"].str[:-2]

# Rename "Location ID" column to "Site"
UV_df.rename(columns={"Location ID": "Site"}, inplace=True)

# Count the number of rows for each site
UV_count = UV_df.groupby('Site').size().reset_index(name='UV_count')

# # Merge data and calculate BC concentrations
# Merge DataFrames
merged_df = pd.merge(UV_df, HIPS_df, on=["FilterID"], how='inner')


# Convert the relevant columns to numeric to handle any non-numeric values
merged_df['BC_HIPS_ug'] = pd.to_numeric(merged_df['BC_HIPS_ug'], errors='coerce')
merged_df['f_BC'] = pd.to_numeric(merged_df['f_BC'], errors='coerce')
merged_df['mass_ug'] = pd.to_numeric(merged_df['mass_ug'], errors='coerce')
merged_df['Volume_m3'] = pd.to_numeric(merged_df['Volume_m3'], errors='coerce')

# Calculate BC concentrations
merged_df['BC_HIPS_(ug/m3)'] = merged_df['BC_HIPS_ug'] / merged_df['Volume_m3']
merged_df['BC_UV-Vis_(ug/m3)'] = merged_df['f_BC'] * merged_df['mass_ug'] / merged_df['Volume_m3']

# Drop the "Site_y" column
merged_df.drop("Site_y", axis=1, inplace=True)
# Rename the "Site_x" column to "Site"
merged_df.rename(columns={"Site_x": "Site"}, inplace=True)

# Write the merged data to separate sheets in an Excel file
with pd.ExcelWriter(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"), engine='openpyxl') as writer:
    # Write the merged data
    merged_df.to_excel(writer, sheet_name='HIPS_UV-Uis', index=False)

    # Print the shape of the merged DataFrame
    # print("Shape of merged_df:", merged_df.shape)

    # Write the HIPS summary sheet
    summary_data = pd.DataFrame(HIPS_count.items(), columns=['Site', 'HIPS'])
    # Add the UV-Vis to the summary data
    summary_data['UV-Vis'] = summary_data['Site'].map(UV_count.set_index('Site')['UV_count'])
    # Sort the data by site name
    summary_data = summary_data.sort_values(by='Site')
    # Write the summary data to a new sheet in the Excel file
    summary_data.to_excel(writer, sheet_name='Summary', index=False)

# Create scatter plot for all sites
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC'] <= 1]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='UV-Vis', y='HIPS', data=merged_df, hue='Site', s=15, alpha=1, ax=ax)
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
plt.title('Comparison of Black Carbon Measured by HIPS and UV-Vis', fontname='Arial', fontsize=16, y=1.03)
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

plt.xlabel('UV-Vis Black Carbon(µg/m$^3$)', fontsize=14, color='black')
plt.ylabel('HIPS Black Carbon (µg/m$^3$)', fontsize=14, color='black')

# show the plot
plt.tight_layout()
plt.savefig(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.tiff"), format="TIFF", dpi=300)
plt.show()