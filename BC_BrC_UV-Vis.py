import numpy as np
import pandas as pd
import os
import shutil

################################################################################################
# Copy R and T from 'Joshin UV Vis Filter Dataset/'
################################################################################################
# Define source and destination root directories
source_root = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Joshin UV Vis Filter Dataset/'
destination_root_r = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Reflectance/'
destination_root_t = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Transmittance/'

# Function to copy files from source to destination directory
def copy_files(source_dir, destination_dir):
    # Create Blank and Acceptance Testing folders inside destination directory if not exists
    blank_folder = os.path.join(destination_dir, "Blank")
    acceptance_testing_folder = os.path.join(destination_dir, "Acceptance Testing")
    if not os.path.exists(blank_folder):
        os.makedirs(blank_folder)
    if not os.path.exists(acceptance_testing_folder):
        os.makedirs(acceptance_testing_folder)
    # Iterate over files in source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().startswith('blank'):  # Check if filename starts with 'blank'
                # Copy file to Blank folder with folder name prefix
                new_filename = f"{subdir}_{file}"
                shutil.copy2(os.path.join(root, file), os.path.join(blank_folder, new_filename))
                print(f"Copied {file} to {blank_folder}")
            elif file.lower().split('.')[0] in ['lb', 'at']:
                # Copy file to Acceptance Testing folder
                shutil.copy2(os.path.join(root, file), acceptance_testing_folder)
                print(f"Copied {file} to {acceptance_testing_folder}")
            else:
                # Copy other files directly to destination directory
                shutil.copy2(os.path.join(root, file), destination_dir)
                print(f"Copied {file} to {destination_dir}")

# Copy files to Reflectance and Transmittance directories
for subdir in os.listdir(source_root):
    subdir_path = os.path.join(source_root, subdir)
    if os.path.isdir(subdir_path):
        for subsubdir in os.listdir(subdir_path):
            subsubdir_path = os.path.join(subdir_path, subsubdir)
            if os.path.isdir(subsubdir_path):
                if "SPARTAN-R-400" in subsubdir:
                    copy_files(subsubdir_path, destination_root_r)
                elif "SPARTAN-T-400" in subsubdir:
                    copy_files(subsubdir_path, destination_root_t)

################################################################################################
# Average together all the files under the "Blank" folder
################################################################################################
# Define the directory paths for reflectance and transmittance blank folders
blank_reflectance_folder = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Reflectance/Blank'
blank_transmittance_folder = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Transmittance/Blank'

def average_blank_files(folder_path, output_file_name):
    # Initialize an empty list to hold DataFrames
    dfs = []

    # Iterate over all files in the folder
    file_count = 0
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
            dfs.append(df)  # Append the DataFrame to the list
            file_count += 1

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs)

    # Get the column name for "%R" or "%T"
    intensity_column = combined_df.columns[1]  # Assuming the second column is intensity

    # Group by the "nm" column and calculate the mean of the intensity column
    averaged_df = combined_df.groupby("nm")[intensity_column].mean().reset_index()

    # Save the averaged DataFrame as a CSV file
    output_file_path = os.path.join(folder_path, output_file_name)
    averaged_df.to_csv(output_file_path, index=False)  # Writing as CSV file without index

    print(f"Number of files averaged: {file_count}")

    return averaged_df, file_count

# Average blank files for reflectance
average_blank_files(blank_reflectance_folder, "Average_Blank_R.csv")

# Average blank files for transmittance
average_blank_files(blank_transmittance_folder, "Average_Blank_T.csv")

################################################################################################
# Calculate MAC and babs
################################################################################################
# Define the directories
r_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Reflectance/'
t_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/Transmittance/'
mass_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'

# Load the blank reflectance and transmittance data
print("Loading blank reflectance and transmittance data...")
r_black_df = pd.read_csv(os.path.join(r_dir, 'Blank', 'Average_Blank_R.csv'))
t_black_df = pd.read_csv(os.path.join(t_dir, 'Blank', 'Average_Blank_T.csv'))
print("Blank reflectance and transmittance data loaded.")

# Load the mass data
mass_dfs = []
for filename in os.listdir(mass_dir):
    if filename.endswith('.csv'):
        print(f"Processing file: {filename}")
        master_data = pd.read_csv(os.path.join(mass_dir, filename), encoding='ISO-8859-1')
        mass_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3',
                        'BC_SSR_ug', 'BC_HIPS_ug']
        if all(col in master_data.columns for col in mass_columns):
            master_data.columns = master_data.columns.str.strip()
            mass_df = master_data[mass_columns].copy()
            mass_df['mass_ug'] = pd.to_numeric(mass_df['mass_ug'], errors='coerce')
            mass_df['Volume_m3'] = pd.to_numeric(mass_df['Volume_m3'], errors='coerce')
            mass_df = mass_df.dropna(subset=['mass_ug'])
            mass_df['PM_conc'] = mass_df['mass_ug'] / mass_df['Volume_m3']
            site_name = filename.split('_')[0]
            mass_df["Site"] = [site_name] * len(mass_df)
            mass_dfs.append(mass_df)
        else:
            print(f"Skipping {filename} because not all required columns are present.")

mass_df = pd.concat(mass_dfs, ignore_index=True)
print("All mass DataFrames concatenated.")

# Create a dictionary to store r_df and t_df for each filter ID
rt_dict = {}

for filename in os.listdir(r_dir):
    if 'Sample.Raw' in filename:
        # Extract site and filter_id from filename
        site = filename[:4].strip()
        filter_id = filename[:9].strip()

        r_df = pd.read_csv(os.path.join(r_dir, filename))

        for filename_t in os.listdir(t_dir):
            if 'Sample.Raw' in filename_t and filename_t.startswith(filter_id):
                t_df = pd.read_csv(os.path.join(t_dir, filename_t))
                rt_dict[filter_id] = (r_df, t_df)
                break  # Exit inner loop once matching t_df is found

# Initialize dfs for MAC and babs
MAC_r_dfs = []
babs_dfs = []

# Define the diameter
d = 25 * 1e-3

# Iterate over each filter ID in mass_df
for idx, row in mass_df.iterrows():
    filter_id = row['FilterID']

    # Get corresponding r_df and t_df from rt_dict
    r_df, t_df = rt_dict.get(filter_id, (None, None))

    # Proceed only if both r_df and t_df are found
    if r_df is not None and t_df is not None:
        # Normalize r based on blanks
        relative_r = r_df.div(r_black_df.values.squeeze(), axis=0)
        relative_r = np.clip(relative_r, None, 0.99999999)

        # Normalize t based on blanks
        t_black_df = t_black_df.reindex(t_df.index)
        relative_t = t_df.div(t_black_df.values.squeeze(), axis=0)
        relative_t = np.clip(relative_t, None, 0.99999999)

        # Calculate optical density
        ODs = np.log((1 - relative_r) / relative_t)
        ODs = np.abs(ODs)

        # Calculate MAC_r and babs
        MAC_r = ((0.48 * (ODs ** 1.32)) / row['mass_ug']) * (np.pi * (d ** 2) / 4) * 1e6
        babs = ((0.48 * (ODs ** 1.32)) / row['mass_ug']) * (np.pi * (d ** 2) / 4) * 1e6

        # Append to lists
        MAC_r_dfs.append(MAC_r)
        babs_dfs.append(babs)

# Concatenate all DataFrames into a single DataFrame
MAC_r_df = pd.concat(MAC_r_dfs, ignore_index=True)
babs_df = pd.concat(babs_dfs, ignore_index=True)
print("All MAC and babs DataFrames concatenated.")

# Calculate other parameters
f_BC = (babs_df[babs_df['nm'] == 900] / mass_df['PM_conc_(ug/m3)']) / 4.58

# Create DataFrame with required columns
result_df = pd.DataFrame({
    'Filter ID': mass_df['FilterID'],
    'mass_ug': mass_df['mass_ug'],
    'PM_conc': mass_df['PM_conc_(ug/m3)'],
    'MAC_r': MAC_r_df.mean(axis=1),
    'babs': babs_df.mean(axis=1),
    'f_BC': f_BC
})
print("Result DataFrame created.")

# Save DataFrame to Excel
excel_filename = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Reflectance/SPARTAN_BC_BrC_UV-Vis.xlsx"
result_df.to_excel(excel_filename, index=False)
################################################################################################
# Summary MAC by site
################################################################################################
import pandas as pd
import numpy as np

# Define the input and output file paths
input_file = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/BC_UV-Vis_SPARTAN_Joshin_20230510.xlsx'
output_file = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/UV-Vis_MAC_Joshin_20230510_Summary.xlsx'

# Load the data from the Excel file
df = pd.read_excel(input_file)

# Filter out rows where 'f_BC' > 0.8
df_filtered = df[df['f_BC'] <= 0.8]

# Group the data by 'Location ID'
grouped = df_filtered.groupby('Location ID')

# Define a function to calculate the standard error
def std_error(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))

# Calculate statistics for MAC columns
summary = grouped.agg({
    'MAC900nm': ['mean', 'median', 'count', std_error],
    'MAC653nm': ['mean', 'median', 'count', std_error],
    'MAC403nm': ['mean', 'median', 'count', std_error]
})

# Flatten the multi-level columns
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

# Save the summary statistics to a new Excel file with "Summary" sheet
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    summary.to_excel(writer, sheet_name='Summary')

print(f'Summary statistics saved to {output_file}')
################################################################################################
# Compare UV-Vis with HIPS by site
################################################################################################
import pandas as pd
import numpy as np
from scipy.stats import linregress

# Define file paths
input_file = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_FTIR_UV-Vis_SPARTAN_allYears.xlsx'
output_file = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_UV-Vis_SPARTAN/UV-Vis_Joshin_20230510_HIPS_ComparionsBySite.xlsx'

# Load the data from the specified sheet
df = pd.read_excel(input_file, sheet_name='All')

# Filter rows where both 'BC_HIPS_ug/m3' and 'BC_UV-Vis_ug/m3' have data (non-null)
df_filtered = df.dropna(subset=['BC_HIPS_ug/m3', 'BC_UV-Vis_ug/m3'])

# Exclude rows where 'f_BC' > 0.8
df_filtered = df_filtered[df_filtered['f_BC'] <= 0.8]
df_filtered = df_filtered[df_filtered['BC_HIPS_ug/m3'] > 0]
# Group by 'Site'
grouped = df_filtered.groupby('Site')

# Create a list to store summary statistics and regression results
summary_list = []

# Loop through each group (site)
for site, group in grouped:
    # Calculate statistics for 'BC_HIPS_ug/m3' and 'BC_UV-Vis_ug/m3'
    hips_avg = group['BC_HIPS_ug/m3'].mean()
    hips_median = group['BC_HIPS_ug/m3'].median()
    hips_std_error = group['BC_HIPS_ug/m3'].std() / np.sqrt(len(group))

    uv_avg = group['BC_UV-Vis_ug/m3'].mean()
    uv_median = group['BC_UV-Vis_ug/m3'].median()
    uv_std_error = group['BC_UV-Vis_ug/m3'].std() / np.sqrt(len(group))

    # Check if there's enough variation in the data for regression
    if group['BC_HIPS_ug/m3'].std() > 0:
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(group['BC_HIPS_ug/m3'], group['BC_UV-Vis_ug/m3'])
        r_squared = r_value ** 2
    else:
        # If no variation, skip regression and set results to NaN
        slope = np.nan
        intercept = np.nan
        r_squared = np.nan
        std_err = np.nan

    # Append the results
    summary_list.append({
        'Site': site,
        'BC_HIPS_Avg': hips_avg,
        'BC_HIPS_Median': hips_median,
        'BC_HIPS_StdError': hips_std_error,
        'BC_UV-Vis_Avg': uv_avg,
        'BC_UV-Vis_Median': uv_median,
        'BC_UV-Vis_StdError': uv_std_error,
        'Slope': slope,
        'Intercept': intercept,
        'R_squared': r_squared,
        'Count': len(group)
    })

# Create a DataFrame from the summary list
summary_df = pd.DataFrame(summary_list)

# Save the summary to an Excel file
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, sheet_name='HIPS_Comparison', index=False)

print(f"Filtered data and analysis results saved to {output_file}, sheet: 'HIPS_Comparison'")

