import os
import pandas as pd

################################################################################################
# Other: Summarize EMEP EC data -1
################################################################################################

other_obs_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/Other_Measurements/'
EMEP_dir = other_obs_dir + '/EMEP_EC_2019_raw/'

# Initialize empty list to store extracted information
all_data = []

# Iterate through each file in the directory
for filename in os.listdir(EMEP_dir):
    # Exclude files starting with '.'
    if not filename.startswith('.'):
        if filename.endswith('.nas'):
            file_path = os.path.join(EMEP_dir, filename)
            with open(file_path, 'r', encoding='latin-1') as file:
                lines = file.readlines()

                # Extract station name
                station_code = next(
                    (line.split(':')[1].strip() for line in lines if line.startswith('Station code:')), '')

                # Search for line starting with 'starttime'
                start_index = None
                for idx, line in enumerate(lines):
                    if line.startswith('starttime'):
                        start_index = idx
                        break

                # If 'starttime' line found
                if start_index is not None:
                    # Extract data from following lines
                    data = [line.split() for line in lines[start_index:]]
                    # Convert each entry to numeric except the line starting with 'starttime'
                    for i, row in enumerate(data):
                        if i != 0:  # Skip the line starting with 'starttime'
                            data[i] = [float(val) for val in row]  # Convert entries to float

                    # Append station name and data to all_data list
                    all_data.append((station_code, data))

# Save data into separate spreadsheets
with pd.ExcelWriter(os.path.join(other_obs_dir, 'EMEP_EC_raw.xlsx'), engine='openpyxl', mode='w') as writer:
    for station_code, data in all_data:
        # Convert data into DataFrame
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name=station_code, index=False, header=False)