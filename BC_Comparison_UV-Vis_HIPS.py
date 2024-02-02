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
# Combine HIPS and UV-Vis dataset, set site as country and city
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
        HIPS_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3', 'BC_SSR_ug', 'BC_HIPS_ug']

        # Check if all required columns are present
        if all(col in master_data.columns for col in HIPS_columns):
            # Remove leading/trailing whitespaces from column names
            master_data.columns = master_data.columns.str.strip()
            # Select the specified columns
            HIPS_df = master_data[HIPS_columns].copy()

            # Convert the relevant columns to numeric to handle any non-numeric values
            HIPS_df['Mass_type'] = pd.to_numeric(HIPS_df['Mass_type'], errors='coerce')
            # Select PM2.5, rows where Mass_type is 1
            HIPS_df = HIPS_df.loc[HIPS_df['Mass_type'] == 1]

            # Convert 'start_year' to numeric and then to integers
            HIPS_df['start_year'] = pd.to_numeric(HIPS_df['start_year'], errors='coerce')
            HIPS_df['start_year'] = HIPS_df['start_year'].astype('Int64')  # 'Int64' allows for NaN values
            # Drop rows with NaN values in the 'start_year' column
            HIPS_df = HIPS_df.dropna(subset=['start_year'])

            # Extract the site name from the filename
            site_name = filename.split('_')[0]
            # Add the site name as a column in the selected data
            HIPS_df["Site"] = [site_name] * len(HIPS_df)

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

# # Merge data and calculate BC mass, concentrations, and fractions
# Merge DataFrames
merged_df = pd.merge(UV_df, HIPS_df, on=["FilterID"], how='inner')

# Convert the relevant columns to numeric to handle any non-numeric values
merged_df['BC_HIPS_ug'] = pd.to_numeric(merged_df['BC_HIPS_ug'], errors='coerce')
merged_df['BC_SSR_ug'] = pd.to_numeric(merged_df['BC_SSR_ug'], errors='coerce')
merged_df['f_BC'] = pd.to_numeric(merged_df['f_BC'], errors='coerce')
merged_df['mass_ug'] = pd.to_numeric(merged_df['mass_ug'], errors='coerce')
merged_df['Volume_m3'] = pd.to_numeric(merged_df['Volume_m3'], errors='coerce')

# Calculate BC mass
merged_df['BC_UV-Vis_ug'] = merged_df['f_BC'] * merged_df['mass_ug']

# Calculate BC concentrations
merged_df['BC_HIPS_(ug/m3)'] = merged_df['BC_HIPS_ug'] / merged_df['Volume_m3']
merged_df['BC_UV-Vis_(ug/m3)'] = merged_df['f_BC'] * merged_df['mass_ug'] / merged_df['Volume_m3']
merged_df['BC_SSR_(ug/m3)'] = merged_df['BC_SSR_ug'] / merged_df['Volume_m3']

# Calculate BC fractions
merged_df.rename(columns={"f_BC": "f_BC_UV-Vis"}, inplace=True)
merged_df['f_BC_HIPS'] = merged_df['BC_HIPS_ug'] / merged_df['mass_ug']

# Drop the "Site_y" column
merged_df.drop("Site_y", axis=1, inplace=True)
# Rename the "Site_x" column to "Site"
merged_df.rename(columns={"Site_x": "Site"}, inplace=True)

# Read Site name from Site_detail.xlsx
Site_df = pd.read_excel(os.path.join(Site_dir_path, 'Site_details.xlsx'), usecols=["Site_Code", "Country", "City"])

# Merge the dataframes based on the "Site" and "Site_Code" columns
merged_df = pd.merge(merged_df, Site_df, how="left", left_on="Site", right_on="Site_Code")

# Drop the duplicate "Site_Code" column
merged_df.drop("Site_Code", axis=1, inplace=True)

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

################################################################################################
# Create scatter plot for all sites
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)
merged_df.rename(columns={"BC_SSR_(ug/m3)": "SSR"}, inplace=True)
merged_df.rename(columns={"City": "city"}, inplace=True)
# Drop rows with NaN values in the 'SSR' column
# merged_df = merged_df.dropna(subset=['SSR'])
# Print the names of each city
unique_cities = merged_df['city'].unique()
for city in unique_cities:
    print(f"City: {city}")

# Classify 'city' based on 'region'
region_mapping = {
    'Central Asia': ['Abu Dhabi', 'Haifa', 'Rehovot'],
    'South Asia': ['Dhaka', 'Bandung', 'Delhi', 'Kanpur', 'Manila', 'Singapore', 'Hanoi'],
    'East Asia': ['Beijing', 'Seoul', 'Ulsan', 'Kaohsiung', 'Taipei'],
    'North America': ['Downsview', 'Halifax', 'Kelowna', 'Lethbridge', 'Sherbrooke', 'Mexico City', 'Fajardo',
                      'Baltimore', 'Bondville', 'Mammoth Cave', 'Norman', 'Pasadena'],
    'South America': ['Buenos Aires', 'Santiago', 'Palmira'],
    'Africa': ['Bujumbura', 'Addis Ababa', 'Ilorin', 'Johannesburg', 'Pretoria'],
    'Australia': ['Melbourne']
}
# Define custom palette for each region with 5 shades for each color
# Define custom palette for each region with 5 shades for each color
region_colors = {
    'Central Asia': [
        (1, 0.42, 0.70), (0.8, 0.52, 0.7), (1, 0.48, 0.41), (1, 0.64, 0.64), (1, 0.76, 0.48)
    ],  # Pink shades
    'South Asia': [
        (0.5, 0, 0), (0.8, 0, 0), (1, 0, 0), (1, 0.2, 0.2), (1, 0.4, 0.4), (1, 0.6, 0.6)
    ],  # Red shades
    'East Asia': [
        (1, 0.64, 0), (1, 0.55, 0.14), (1, 0.63, 0.48), (1, 0.74, 0.61), (1, 0.85, 0.73), (1, 0.96, 0.85)
    ],  # Orange shades
    'North America': [
        (0, 0, 0.5), (0, 0, 0.8), (0, 0, 1), (0.39, 0.58, 0.93), (0.54, 0.72, 0.97), (0.68, 0.85, 0.9)
    ],  # Blue shades
    'South America': [
        (0.58, 0.1, 0.81), (0.9, 0.4, 1), (0.66, 0.33, 0.83), (0.73, 0.44, 0.8), (0.8, 0.55, 0.77), (0.88, 0.66, 0.74)
    ],  # Purple shades
    'Africa': [
        (0, 0.5, 0), (0, 0.8, 0), (0, 1, 0), (0.56, 0.93, 0.56), (0.56, 0.93, 0.56), (0.8, 0.9, 0.8)
    ],  # Green shades
    'Australia': [
        (0.6, 0.4, 0.2)
    ]  # Brown
}

def map_city_to_color(city):
    for region, cities in region_mapping.items():
        if city in cities:
            city_index = cities.index(city) % len(region_colors[region])
            assigned_color = region_colors[region][city_index]
            print(f"City: {city}, Region: {region}, Assigned Color: {assigned_color}")
            return assigned_color
    print(f"City not found in any region: {city}")
    return (0, 0, 0)  # Default to black if city is not found
# Create an empty list to store the city_palette for each city
city_palette = []
city_color_match = []
# Iterate over each unique city and map it to a gradient
for city in unique_cities:
    city_color = map_city_to_color(city)
    if city_color is not None:
        city_palette.append(city_color)
        city_color_match.append({'city': city, 'color': city_color})  # Store both city name and color
print("City Palette:", city_palette)

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))
# fig, ax = plt.subplots(figsize=(8, 7)) # without legend
# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='SSR', y='HIPS', data=merged_df, hue='city', palette=city_palette, s=50, alpha=1, ax=ax, edgecolor='k')
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Create a function to determine the index of a city in region_mapping
def get_city_index(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return cities.index(city)
    return float('inf')  # If city is not found, place it at the end
def get_region_for_city(city):
    for region, cities in region_mapping.items():
        if city in cities:
            return region
    print(f"Region not found for city: {city}")
    return None
# Sort the unique_cities list based on their appearance in region_mapping
unique_cities_sorted = sorted(unique_cities, key=get_city_index)
# Create legend with custom order
sorted_city_color_match = sorted(city_color_match, key=lambda x: (
    list(region_mapping.keys()).index(get_region_for_city(x['city'])),
    region_mapping[get_region_for_city(x['city'])].index(x['city'])
))

legend_labels = [city['city'] for city in sorted_city_color_match]
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=city['color'], markersize=8, label=city['city']) for city in sorted_city_color_match]
legend = plt.legend(handles=legend_handles, facecolor='white', bbox_to_anchor=(1.03, 0.50), loc='center left', fontsize=10)
# legend.get_frame().set_linewidth(0.0)


# Set title, xlim, ylim, ticks, labels
plt.title('Comparison of Black Carbon Concentration Measured by HIPS and SSR', fontname='Arial', fontsize=16, y=1.03)
plt.xlim([merged_df['HIPS'].min()-0.5, 35])
plt.ylim([merged_df['HIPS'].min()-0.5, 35])
plt.xticks([0, 10, 20, 30], fontname='Arial', size=18)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=18)
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
mask = ~np.isnan(merged_df['SSR']) & ~np.isnan(merged_df['HIPS'])
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['SSR'][mask], merged_df['HIPS'][mask])
# Check for NaN in results
if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
    print("Linear regression results contain NaN values. Check the input data.")
else:
    # Add linear regression line and text
    sns.regplot(x='SSR', y='HIPS', data=merged_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
    # Change the sign of the intercept for display
    intercept_display = abs(intercept)  # Use abs() to ensure a positive value
    intercept_sign = '-' if intercept < 0 else '+'  # Determine the sign for display

    # Update the text line with the adjusted intercept
    plt.text(0.1, 0.76, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
             transform=plt.gca().transAxes, fontsize=22)

plt.xlabel('SSR Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')
plt.ylabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(os.path.join(Out_dir_path, "BC_Concentration_Comparison_HIPS_SSR.tiff"), format="TIFF", dpi=300)
plt.show()

################################################################################################
# Create scatter plot for each site
################################################################################################

# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]

# Rename to simplify coding
merged_df.rename(columns={"BC_HIPS_(ug/m3)": "HIPS"}, inplace=True)
merged_df.rename(columns={"BC_UV-Vis_(ug/m3)": "UV-Vis"}, inplace=True)

# Get a list of all unique site names
site_names = merged_df['City'].unique()

# Set the number of columns for the combined plot
num_columns = 7

# Calculate the number of rows needed
num_rows = int(np.ceil(len(site_names) / num_columns))

# Create a figure and axis with the specified layout
fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 2 * num_rows))

# Flatten the axes array to simplify iteration
axes = axes.flatten()

# Loop through each subplot and plot the data for the corresponding site
for i, site_name in enumerate(site_names):
    # Get the data for the current site
    site_data = merged_df.loc[merged_df['City'] == site_name]

    # Skip site if no data points
    if len(site_data) < 2:
        continue

    # Create scatter plot on the current subplot
    scatterplot = sns.scatterplot(x='UV-Vis', y='HIPS', data=site_data, s=15, alpha=1, ax=axes[i])
    scatterplot.set_facecolor('white')
    border_width = 1
    for spine in scatterplot.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(border_width)
    scatterplot.grid(False)

    # Set axis limits for the current subplot
    x_min, x_max = site_data[['UV-Vis', 'HIPS']].min().min() - 0.2, site_data[['UV-Vis', 'HIPS']].max().max() + 0.5
    y_min, y_max = site_data[['UV-Vis', 'HIPS']].min().min() - 0.2, site_data[['UV-Vis', 'HIPS']].max().max() + 0.5
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)

    # Calculate reasonable tick intervals based on the range of data
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_ticks_interval = get_tick_interval(x_range)
    y_ticks_interval = get_tick_interval(y_range)

    # Set x and y ticks with calculated interval
    scatterplot.xaxis.set_major_locator(MultipleLocator(x_ticks_interval))
    scatterplot.yaxis.set_major_locator(MultipleLocator(y_ticks_interval))

    # Set x and y tick labels with a specific number of digits
    scatterplot.set_xticks(scatterplot.get_xticks())
    scatterplot.set_xticklabels([f'{tick:.2f}' for tick in scatterplot.get_xticks()],
                                fontdict={'fontname': 'Arial', 'fontsize': 8})
    scatterplot.set_yticks(scatterplot.get_yticks())
    scatterplot.set_yticklabels([f'{tick:.2f}' for tick in scatterplot.get_yticks()],
                                fontdict={'fontname': 'Arial', 'fontsize': 8})

    # Add 1:1 line with grey dash
    axes[i].plot([x_min, x_max], [y_min, y_max], color='grey', linestyle='--', linewidth=1)

    # Add the number of data points to the plot
    num_points = len(site_data)
    axes[i].text(0.05, 0.65, f'N = {num_points}', transform=axes[i].transAxes, fontsize=8)

    # Perform linear regression with NaN handling
    mask = ~np.isnan(site_data['UV-Vis']) & ~np.isnan(site_data['HIPS'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(site_data['UV-Vis'][mask], site_data['HIPS'][mask])

    # Check for NaN in results
    if np.isnan(slope) or np.isnan(intercept) or np.isnan(r_value):
        print("Linear regression results contain NaN values. Check the input data.")
    else:
        # Add linear regression line and text
        sns.regplot(x='UV-Vis', y='HIPS', data=site_data, scatter=False, ci=None,
                    line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1}, ax=axes[i])

        # Change the sign of the intercept for display
        intercept_display = abs(intercept)
        intercept_sign = '-' if intercept < 0 else '+'
        axes[i].text(0.05, 0.75, f"y = {slope:.2f}x {intercept_sign} {intercept_display:.2f}\n$r^2$ = {r_value ** 2:.2f}",
                     transform=axes[i].transAxes, fontsize=8)

    # Set plot title as site name
    axes[i].set_title(site_name, fontname='Arial', fontsize=10)

    # Set x and y labels for the subplot
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    # axes[i].set_xlabel('UV-Vis Black Carbon Concentration (µg/m$^3$)', fontsize=8, fontname='Arial')
    # axes[i].set_ylabel('HIPS Black Carbon Concentration (µg/m$^3$)', fontsize=8, fontname='Arial')

# set x and y labels above the subplots
fig.text(0.5, 0.05, 'UV-Vis Black Carbon Concentration (µg/m$^3$)', ha='center', va='center', fontsize=14, fontname='Arial')
fig.text(0.03, 0.5, 'HIPS Black Carbon Concentration (µg/m$^3$)', ha='center', va='center', rotation='vertical', fontsize=14, fontname='Arial')

# Adjust vertical distance among subplots
fig.tight_layout()

# Adjust the figure size to accommodate the common x and y labels
fig.subplots_adjust(bottom=0.12, left=0.06)

# Save the combined plot as an image (optional)
# plt.savefig(os.path.join(Out_dir_path, "BC_Concentration_Comparison_by_Site.tiff"), format="tiff", dpi=300)
plt.show()

################################################################################################
# Investigate diff in BC fraction vs PM mass
################################################################################################
# Read the file
merged_df = pd.read_excel(os.path.join(Out_dir_path, "BC_Comparison_HIPS_UV-Vis.xlsx"))

# Convert 'start_year', 'start_month', and 'start_day' to datetime format
merged_df['start_date'] = pd.to_datetime(merged_df[['start_year', 'start_month', 'start_day']].astype(str).agg('/'.join, axis=1), errors='coerce')
# print(merged_df[['start_date']].sort_values(by='start_date'))

# Calculate the difference in BC fraction and mass
merged_df['diff'] = merged_df['f_BC_UV-Vis'] - merged_df['f_BC_HIPS']
merged_df['diff_ug'] = merged_df['BC_UV-Vis_ug'] - merged_df['BC_HIPS_ug']

# Drop rows where f_BC is greater than 1
merged_df = merged_df.loc[merged_df['f_BC_UV-Vis'] <= 1]

# Drop rows where mass_ug is greater than 200
# merged_df = merged_df.loc[merged_df['mass_ug'] <= 200]

# Rename to simplify coding
# merged_df.rename(columns={"diff_ug": "diff"}, inplace=True)   # only use in bc mass
merged_df.rename(columns={"mass_ug": "mass"}, inplace=True)
merged_df.rename(columns={"start_date": "date"}, inplace=True)

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

# Set x-axis limits with a buffer
# buffer_days = 20
# plt.xlim([merged_df['date'].min() - DateOffset(days=buffer_days), merged_df['date'].max() + DateOffset(days=buffer_days)])

# Set title, xlim, ylim, ticks, labels
plt.title('Difference in Black Carbon Fraction Measured by UV-Vis and HIPS v.s. PM Mass', fontname='Arial', fontsize=14, y=1.03)
plt.xlim([merged_df['mass'].min() - 5, merged_df['mass'].max() + 5])
plt.ylim([merged_df['diff'].min() - 0.05, merged_df['diff'].max() + 0.05])
plt.xticks(fontname='Arial', size=14)
plt.yticks(fontname='Arial', size=14)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add reference line at y = 0
ax.axhline(y=0, color='grey', linestyle='--', linewidth=1)

# Add number of data points to the plot
num_points = len(merged_df)
plt.text(0.05, 0.80, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)
plt.xlabel('PM Mass (µg)', fontsize=14, color='black', fontname='Arial')
plt.ylabel('Δ Black Carbon Fraction (UV-Vis - HIPS)', fontsize=14, color='black', fontname='Arial')

# show the plot
plt.tight_layout()
# plt.savefig(os.path.join(Out_dir_path, "Diff_BC_Fraction_PM_Mass.tiff"), format="TIFF", dpi=300)
plt.show()