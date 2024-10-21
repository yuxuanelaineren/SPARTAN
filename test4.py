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
