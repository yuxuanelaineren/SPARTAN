import os
import pandas as pd

other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
EMEP_dir = other_obs_dir + '/EMEP_EC_2019_raw/'

# Initialize empty lists to store extracted information
EC_df = []

# Iterate through each file in the directory
for filename in os.listdir(EMEP_dir):
    if filename.endswith('.nas'):
        file_path = os.path.join(EMEP_dir, filename)
        with open(file_path, 'r', encoding='latin-1') as file:
            lines = file.readlines()

            # Extract required information
            station_code = next((line.split(':')[1].strip() for line in lines if line.startswith('Station code:')), "")
            station_name = next((line.split(':')[1].strip() for line in lines if line.startswith('Station name:')), "")
            latitude = next((line.split(':')[1].strip() for line in lines if line.startswith('Station latitude:')), "")
            longitude = next((line.split(':')[1].strip() for line in lines if line.startswith('Station longitude:')), "")
            component = next((line.split(':')[1].strip() for line in lines if line.startswith('Component:')), "")
            unit = next((line.split(':')[1].strip() for line in lines if line.startswith('Unit:')), "")
            analytical_measurement_technique = next((line.split(':')[1].strip() for line in lines if line.startswith('Analytical measurement technique:')), "")
            data_level = next((line.split(':')[1].strip() for line in lines if line.startswith('Data level:')), "")
            # print(station_name, latitude, longitude)

            # Append extracted information to EC_df list
            EC_df.append([station_code, station_name, latitude, longitude, component, unit, analytical_measurement_technique, data_level])

# Convert the list of lists into a DataFrame
df = pd.DataFrame(EC_df, columns=['Station code', 'Station Name', 'Latitude', 'Longitude', 'Component', 'Unit', 'Analytical_Measurement_Technique', 'data_level'])

# Save the DataFrame as an Excel file
with pd.ExcelWriter(os.path.join(other_obs_dir, 'EMEP_EC_Summary.xlsx'), engine='openpyxl', mode='w') as writer:
    df.to_excel(writer, index=False)
