import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the input Excel file
obs_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/Fe_BC_HIPS_SPARTAN.xlsx',
                       sheet_name='All')

# Group by 'Site', 'Country', 'City', and 'start_month', and calculate monthly mean, median, and count for each column
obs_monthly_df = obs_df.groupby(['Site', 'Country', 'City', 'start_month']).agg(
    monthly_mean_BC=('BC', 'mean'),
    monthly_median_BC=('BC', 'median'),
    monthly_count_BC=('BC', 'count'),
    monthly_mean_Fe=('Fe', 'mean'),
    monthly_median_Fe=('Fe', 'median'),
    monthly_count_Fe=('Fe', 'count'),
    monthly_mean_Fe_BC=('Fe_BC', 'mean'),
    monthly_median_Fe_BC=('Fe_BC', 'median'),
    monthly_count_Fe_BC=('Fe_BC', 'count'),
    monthly_mean_Fe_PM25=('Fe_PM25', 'mean'),
    monthly_median_Fe_PM25=('Fe_PM25', 'median'),
    monthly_count_Fe_PM25=('Fe_PM25', 'count')
).reset_index()

# Calculate the annual averages for each 'Site', 'Country', 'City' based on the monthly values
obs_annual_df = obs_monthly_df.groupby(['Site', 'Country', 'City']).agg(
    annual_mean_BC=('monthly_mean_BC', 'mean'),
    annual_median_BC=('monthly_median_BC', 'median'),
    annual_count_BC=('monthly_count_BC', 'sum'),  # Count for the entire year
    annual_se_BC=('monthly_mean_BC', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),  # SE = std / sqrt(n)

    annual_mean_Fe=('monthly_mean_Fe', 'mean'),
    annual_median_Fe=('monthly_median_Fe', 'median'),
    annual_count_Fe=('monthly_count_Fe', 'sum'),
    annual_se_Fe=('monthly_mean_Fe', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),

    annual_mean_Fe_BC=('monthly_mean_Fe_BC', 'mean'),
    annual_median_Fe_BC=('monthly_median_Fe_BC', 'median'),
    annual_count_Fe_BC=('monthly_count_Fe_BC', 'sum'),
    annual_se_Fe_BC=('monthly_mean_Fe_BC', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),  # SE = std / sqrt(n)

    annual_mean_Fe_PM25=('monthly_mean_Fe_PM25', 'mean'),
    annual_median_Fe_PM25=('monthly_median_Fe_PM25', 'median'),
    annual_count_Fe_PM25=('monthly_count_Fe_PM25', 'sum'),
    annual_se_Fe_PM25=('monthly_mean_Fe_PM25', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # SE = std / sqrt(n)
).reset_index()

# Save the results to Excel with separate sheets for monthly and annual data
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/Fe_BC_HIPS_SPARTAN.xlsx',
                    engine='openpyxl', mode='a') as writer:
    obs_monthly_df.to_excel(writer, sheet_name='Mon', index=False)
    obs_annual_df.to_excel(writer, sheet_name='Annual', index=False)
