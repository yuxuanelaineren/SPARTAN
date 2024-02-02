import numpy as np
import pandas as pd
import os
import shutil

################################################################################################
# Copy R and T from 'Joshin UV Vis Filter Dataset/'
################################################################################################

# Define source and destination root directories
source_root = "/Users/renyuxuan/Desktop/Research/Black_Carbon/Joshin UV Vis Filter Dataset/"
destination_root_r = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Reflectance/"
destination_root_t = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Transmittance/"

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
blank_reflectance_folder = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Reflectance/Blank"
blank_transmittance_folder = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Transmittance/Blank"

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
average_blank_files(blank_reflectance_folder, "Average_BLANK_R.csv")

# Average blank files for transmittance
average_blank_files(blank_transmittance_folder, "Average_BLANK_T.csv")

################################################################################################
# Calculate MAC and babs
################################################################################################

# Define the directories
r_dir = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Reflectance/"
t_dir = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Transmittance/"
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
            mass_df['PM_conc_(ug/m3)'] = mass_df['mass_ug'] / mass_df['Volume_m3']
            site_name = filename.split('_')[0]
            mass_df["Site"] = [site_name] * len(mass_df)
            mass_dfs.append(mass_df)
        else:
            print(f"Skipping {filename} because not all required columns are present.")

mass_df = pd.concat(mass_dfs, ignore_index=True)
print("All mass DataFrames concatenated.")

# Create a dictionary to store r_df and t_df for each filter ID
rt_dict = {}

# Populate the dictionary with r_df and t_df for each filter ID
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

# Initialize lists for MAC and babs
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
        # Calculate relative reflectance
        relative_r = r_df.div(r_black_df.values.squeeze(), axis=0)
        relative_r = np.clip(relative_r, None, 0.99999999)

        # Calculate relative transmittance
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
