import numpy as np
import pandas as pd
import os
import shutil

################################################################################################
# Calculate MAC and babs
################################################################################################

# Define the directories
r_dir = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/ETAD/Reflectance/"
t_dir = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/ETAD/Transmittance/"
mass_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'

# Load the blank reflectance and transmittance data
print("Loading blank reflectance and transmittance data...")
r_black_df = pd.read_csv(os.path.join(r_dir, 'Blank', 'Average_Blank_R.csv'))
t_black_df = pd.read_csv(os.path.join(t_dir, 'Blank', 'Average_Blank_T.csv'))
print("Blank reflectance and transmittance data loaded.")

# Load the mass data
mass_dfs = []
filename = 'ETAD_master.csv'

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
print(mass_df.head())

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

# Initialize for MAC and babs
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
        print(relative_r.head())
        # Calculate optical depth
        ODs = np.log((1 - relative_r) / relative_t)
        ODs = np.abs(ODs)
        print(ODs.head())
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
# print(MAC_r_df.head())
# print(babs_df.head())

# Calculate other parameters
f_BC = (babs_df[babs_df['nm'] == 900] / mass_df['PM_conc_(ug/m3)']) / 4.58
f_BC = f_BC.mean(axis=1)  # Calculate mean along rows

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
excel_filename = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/ETAD/SPARTAN_BC_BrC_UV-Vis_ETAD.xlsx"
result_df.to_excel(excel_filename, index=False)