import os
import pandas as pd

# Set the directory path
HIPS_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
Site_dir_path = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
Out_dir_path = '/Users/renyuxuan/Desktop/Research/Black_Carbon/UV-Vis_HIPS_Comparison/'


################################################################################################
# Calculate statistics for HIPS dataset, set site as country and city
################################################################################################




################################################################################################
# Calculate statistics for HIPS dataset, set site as country and city
################################################################################################
# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Create an empty dictionary to store the row count by site name
HIPS_count = {}

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

# # Calculate summary statistics for BC_HIPS_(ug/m3)
# Convert the relevant columns to numeric to handle any non-numeric values
HIPS_df['BC_HIPS_ug'] = pd.to_numeric(HIPS_df['BC_HIPS_ug'], errors='coerce')
HIPS_df['mass_ug'] = pd.to_numeric(HIPS_df['mass_ug'], errors='coerce')
HIPS_df['Volume_m3'] = pd.to_numeric(HIPS_df['Volume_m3'], errors='coerce')

# Calculate BC concentrations
HIPS_df['BC_HIPS_(ug/m3)'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['Volume_m3']
HIPS_df['f_BC_HIPS'] = HIPS_df['BC_HIPS_ug'] / HIPS_df['mass_ug']

# Create an ExcelWriter to write to the same Excel file
with pd.ExcelWriter(os.path.join(Out_dir_path, "HIPS_SPARTAN.xlsx"), engine='openpyxl') as writer:
    # Write the HIPS data to the 'HIPS_UV-Vis' sheet
    HIPS_df.to_excel(writer, sheet_name='HIPS_All', index=False)

    # Calculate summary statistics for BC_HIPS_(ug/m3) before merging
    HIPS_stats = HIPS_df.groupby('Site').agg({
        'BC_HIPS_(ug/m3)': ['mean', 'std', 'max', 'min'],
        'f_BC_HIPS': ['mean', 'std', 'max', 'min']
    }).reset_index()

    # Flatten multi-level column names
    HIPS_stats.columns = ['_'.join(col).strip() for col in HIPS_stats.columns.values]

    # Rename columns for clarity
    HIPS_stats.rename(columns={
        'Site_': 'Site',
        'BC_HIPS_(ug/m3)_mean': 'BC_HIPS_mean',
        'BC_HIPS_(ug/m3)_std': 'BC_HIPS_std',
        'BC_HIPS_(ug/m3)_max': 'BC_HIPS_max',
        'BC_HIPS_(ug/m3)_min': 'BC_HIPS_min',
        'f_BC_HIPS_mean': 'f_BC_HIPS_mean',
        'f_BC_HIPS_std': 'f_BC_HIPS_std',
        'f_BC_HIPS_max': 'f_BC_HIPS_max',
        'f_BC_HIPS_min': 'f_BC_HIPS_min',
    }, inplace=True)

    # Add the count to the statis data
    number_df = pd.DataFrame(list(HIPS_count.items()), columns=['Site', 'Number'])
    HIPS_stats = pd.merge(HIPS_stats, number_df, on='Site')

    # Write the summary statistics to the 'HIPS_All_Summary' sheet
    HIPS_stats.to_excel(writer, sheet_name='HIPS_All_Summary', index=False)



