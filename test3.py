
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  # cartopy must be >=0.19
import xarray as xr
import cartopy.feature as cfeature
import pandas as pd
from gamap_colormap import WhGrYlRd
import calendar
import numpy as np
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from matplotlib import font_manager
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy.io import loadmat
import matplotlib.lines as mlines
from scipy.stats import linregress
cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-EDGARv61-vert-{}-output/monthly/'.format(cres.lower(), deposition) # EDGAR, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-HTAPv3-vert-{}-output/monthly/'.format(cres.lower(), deposition) # HTAP, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-CSwinds-output/monthly/'.format(cres.lower(), deposition) # CEDS, C3720, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C360, LUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, GEOS-FP
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/{}-CEDS01-fixed-vert-{}-merra2-output/monthly/'.format(cres.lower(), deposition) # CEDS, C180, noLUO, MERRA2
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
site_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Site_Sampling/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)
support_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/supportData/'
otherMeas_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/otherMeasurements/'
################################################################################################
# Effects of COVID-19 lockdown
################################################################################################
# # Load the dataset
# df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='All')
#
# # Define affected (Jan 2020 to June 2021) and non-affected periods
# affected_period = ((df['start_year'] == 2020) | ((df['start_year'] == 2021) & (df['start_month'] <= 6)))
# non_affected_period = ((df['start_year'] == 2019) | ((df['start_year'] == 2021) & (df['start_month'] > 6)) | (df['start_year'] > 2021))
# # affected_period = ((df['start_year'] == 2020) | (df['start_year'] == 2021))
# # non_affected_period = ((df['start_year'] == 2019) | (df['start_year'] > 2021))
#
# # Count filters by city for each period
# affected_counts = df[affected_period].groupby('City').size().reset_index(name='Affected_Period_Count')
# non_affected_counts = df[non_affected_period].groupby('City').size().reset_index(name='Non_Affected_Period_Count')
# total_counts = df.groupby('City').size().reset_index(name='Total_Count')
#
# # Merge results into a single dataframe
# result = total_counts.merge(affected_counts, on='City', how='left')
# result = result.merge(non_affected_counts, on='City', how='left')
# result = result.fillna(0)
# print(result)
#
# # Merge the affected, non-affected, and total counts by 'City'
# merged_counts = total_counts.merge(affected_counts, on='City', how='left') \
#                             .merge(non_affected_counts, on='City', how='left') \
#                             .fillna(0)
# merged_counts['Non_Affected_Percentage'] = merged_counts['Non_Affected_Period_Count'] / merged_counts['Total_Count']
# print(merged_counts[['City', 'Non_Affected_Percentage']])
#
# # Optional: Save the merged counts to a CSV file
# # merged_counts.to_csv("filter_counts_by_city.csv", index=False)
# # Extract non-affected samples based on the 'non_affected_period'
# non_affected_df = df[non_affected_period]
# # Save the annual DataFrame as 'annual'
# with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', engine='openpyxl', mode='a') as writer:
#     non_affected_df.to_excel(writer, sheet_name='not-affected-by-COVID_All', index=False)
#
# # Group by 'Local Site Name', 'Month', 'Latitude', and 'Longitude'
# obs_monthly_df = non_affected_df.groupby(['Site', 'Country',	'City', 'start_month']).agg(
#     monthly_mean=('BC', 'mean'),
#     monthly_median=('BC', 'median'),
#     monthly_count=('BC', 'count')
# ).reset_index()
#
# # Calculate the annual average 'ECf_Val' for each 'SiteName', 'Latitude', 'Longitude'
# obs_annual_df = obs_monthly_df.groupby(['Site', 'Country',	'City']).agg(
#     annual_mean=('monthly_mean', 'mean'),
#     annual_median=('monthly_median', 'median'),
#     annual_count=('monthly_count', 'sum'),
#     annual_se=('monthly_mean', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)))  # SE = std / sqrt(n)
# ).reset_index()
#
# # Save the annual DataFrame as 'annual'
# with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx', engine='openpyxl', mode='a') as writer:
#     obs_monthly_df.to_excel(writer, sheet_name='non_affected_Mon', index=False)
#     obs_annual_df.to_excel(writer, sheet_name='non_affected_Annual', index=False)
# Example dataset for simulated and measured concentrations (replace with actual data)
data = {
    "City": ["Abu Dhabi", "Melbourne", "Dhaka", "Bujumbura", "Halifax", "Sherbrooke", "Beijing", "Addis Ababa", "Bandung",
             "Haifa", "Rehovot", "Kanpur", "Seoul", "Ulsan", "Mexico City", "Ilorin", "Fajardo", "Kaohsiung", "Taipei",
             "Pasadena", "Johannesburg", "Pretoria"],
    "Csim": [2.603483532, 0.431163175, 4.747680126, 3.673715311, 0.224381786, 0.362798662, 1.385444595, 4.799646778,
             4.02492436, 0.845562015, 1.169340151, 3.833072212, 1.175011504, 0.7798648, 2.008797088, 2.326521987, 0.10290891,
             1.33695288, 0.830166517, 0.474454487, 2.381180572, 2.013390368], # affected by Covid
    "Cmeas": [2.673810294, 0.431163175, 5.56315254, 3.673715311, 0.23148047, 0.363877719, 1.398329746, 4.799646778, 3.663149692,
             0.845562015, 1.159011749, 3.833072212, 1.196440665, 0.7798648, 2.073496912, 2.982349549, 0.10684992, 1.33695288,
             0.830166517, 0.474454487, 2.381180572, 2.098747274] # full dataset
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate the Normalized Mean Bias (NMB)
nmb = np.sum(df['Csim'] - df['Cmeas']) / np.sum(df['Cmeas'])

# Calculate the Normalized Mean Difference (NMD)
nmd = np.sum(np.abs(df['Csim'] - df['Cmeas'])) / np.sum(df['Cmeas'])

# Print the results
print(f"Normalized Mean Bias (NMB): {nmb}")
print(f"Normalized Mean Difference (NMD): {nmd}")
