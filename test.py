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