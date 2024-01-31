import numpy as np
import pandas as pd
import os

# Define the directories
r_dir = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Reflectance/"
t_dir = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Transmittance/"
mass_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'

# Load the blank reflectance and transmittance data
r_black_df = pd.read_csv(os.path.join(r_dir, 'Blank', 'Average_Blank_R.csv'))
t_black_df = pd.read_csv(os.path.join(t_dir, 'Blank', 'Average_Blank_T.csv'))

# Extract PM_mass from the master file folder
mass_dfs = []
mass_count = {}

# Iterate over each file in the directory
for filename in os.listdir(mass_dir):
    if filename.endswith('.csv'):
        # Read the data from the master file
        master_data = pd.read_csv(os.path.join(mass_dir, filename), encoding='ISO-8859-1')

        # Specify the required columns
        mass_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3', 'BC_SSR_ug', 'BC_HIPS_ug']

        # Check if all required columns are present
        if all(col in master_data.columns for col in mass_columns):
            # Remove leading/trailing whitespaces from column names
            master_data.columns = master_data.columns.str.strip()
            # Select the specified columns
            mass_df = master_data[mass_columns].copy()

            # Convert the relevant columns to numeric to handle any non-numeric values
            mass_df['mass_ug'] = pd.to_numeric(mass_df['mass_ug'], errors='coerce')
            mass_df['Volume_m3'] = pd.to_numeric(mass_df['Volume_m3'], errors='coerce')
            # Drop rows with NaN values in the 'mass_ug' column
            mass_df = mass_df.dropna(subset=['mass_ug'])

            # Calculate PM concentrations
            mass_df['PM_conc_(ug/m3)'] = mass_df['mass_ug'] / mass_df['Volume_m3']

            # Extract the site name from the filename
            site_name = filename.split('_')[0]
            # Add the site name as a column in the selected data
            mass_df["Site"] = [site_name] * len(mass_df)

            # Count the number of rows
            row_count = mass_df.shape[0]
            # Add the row count to the dictionaries
            if site_name in mass_count:
                mass_count[site_name] += row_count
            else:
                mass_count[site_name] = row_count

            # Append the current mass_df to the list if the site is "ETAD"
            if site_name == "ETAD":
                mass_dfs.append(mass_df)
        else:
            print(f"Skipping {filename} because not all required columns are present.")

# Concatenate all mass DataFrames for "ETAD" site into a single DataFrame
mass_df = pd.concat(mass_dfs, ignore_index=True)

# Load the lambda values
Lambda_values = pd.read_csv(os.path.join(r_dir, 'Blank', 'Average_Blank_R.csv'))
Lambda_values = Lambda_values.values.flatten()

# Define the diameter
d = 25 * 1e-3

# Initialize lists for MAC and babs
MAC_r_dfs = []
babs_dfs = []

# Iterate over each CSV file containing "Sample.Raw" in the reflectance directory
for filename in os.listdir(r_dir):
    if 'ETAD' in filename:
        # Extract site and filter_id from filename
        site = filename[:4]
        filter_id = filename[:9]

        # Load the reflectance data
        r_df = pd.read_csv(os.path.join(r_dir, filename))

        # Calculate the relative reflectance
        relative_r = r_df.div(r_black_df.values.squeeze(), axis=0)
        relative_r = np.clip(relative_r, None, 0.99999999)  # Clip values to avoid zero or negative values inside log

        # Iterate over each CSV file containing "Sample.Raw" in the transmittance directory
        for filename_t in os.listdir(t_dir):
            if 'ETAD' in filename_t:
                # Load the transmittance data
                t_df = pd.read_csv(os.path.join(t_dir, filename_t))

                # Align the indices of t_df and t_black_df
                t_black_df = t_black_df.reindex(t_df.index)

                # Perform division operation
                relative_t = t_df.div(t_black_df.values.squeeze(), axis=0)
                relative_t = np.clip(relative_t, None, 0.99999999)  # Clip values to avoid zero or negative values inside log

                # Calculate the OD
                ODs = np.log((1 - relative_r) / relative_t)
                ODs = np.abs(ODs)

                # Iterate over mass_df to match filter id
                for idx, row in mass_df.iterrows():
                    if row['FilterID'] == filter_id:
                        # Calculate MAC_r and babs
                        MAC_r = ((0.48 * (ODs ** 1.32)) / row['mass_ug']) * (np.pi * (d ** 2) / 4) * 1e6
                        babs = ((0.48 * (ODs ** 1.32)) / row['mass_ug']) * (np.pi * (d ** 2) / 4) * 1e6

                        # Append the current mass_df to the list
                        MAC_r_dfs.append(MAC_r)  # Fixed the variable name
                        babs_dfs.append(babs)  # Fixed the variable name

# Concatenate all DataFrames into a single DataFrame
MAC_r_df = pd.concat(MAC_r_dfs, ignore_index=True)
babs_df = pd.concat(babs_dfs, ignore_index=True)

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

# Save DataFrame to Excel
excel_filename = "/Users/renyuxuan/Desktop/Research/Black_Carbon/BC_BrC_UV-Vis/Reflectance/ETAD_BC_BrC_UV-Vis.xlsx"
result_df.to_excel(excel_filename, index=False)
