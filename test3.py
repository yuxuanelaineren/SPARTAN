import pandas as pd

cres = 'C360'
year = 2019
species = 'BC'
inventory = 'CEDS'
deposition = 'noLUO'

# Set the directory path
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}/monthly/'.format(cres.lower()) # CEDS, noLUO
# sim_dir = '/Volumes/rvmartin2/Active/Shared/dandan.z/GCHP-v13.4.1/output-{}-noLUO/monthly/'.format(cres.lower(), deposition) # HTAP, LUO
# sim_dir = '/Volumes/rvmartin/Active/dandan.z/AnalData/WUCR3-C360/' # EDGAR, LUO
sim_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/WUCR3-C360/' # EDGAR, LUO
obs_dir = '/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/'
out_dir = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/{}_{}_{}_{}/'.format(cres.lower(), inventory, deposition, year)

################################################################################################
# Calculate monthly and annual average across different years
################################################################################################
# Load the data
obs_df = pd.read_excel(out_dir + 'HIPS_SPARTAN.xlsx')
CEDS_df = pd.read_excel(out_dir + 'C360_CEDS_noLUO_Sim_vs_SPARTAN_BC_2019_Summary.xlsx', sheet_name='Mon',
                        usecols=['lat', 'lon', 'month', 'country', 'city', 'sim'])

# Group by 'start_year' and 'start_month', calculate average and count
obs_df = obs_df.groupby(['start_year', 'start_month', 'Country', 'City'])['BC_HIPS_ug'].agg(['mean', 'count']).reset_index()
obs_df.columns = ['year_obs', 'month', 'country', 'city', 'obs', 'num_obs']

# Merge observation and simulation data
compr_df = pd.merge(obs_df, CEDS_df, on=['month', 'country', 'city'], how='inner')

# Select the desired columns from the merged DataFrame
compr_df = compr_df[['lat', 'lon', 'country', 'city', 'year_obs', 'month', 'sim', 'obs', 'num_obs']]

# compr_df = compr_df[obs_df['city'] == 'Beijing']
# Calculate annual average for each site
annual_df = compr_df.groupby(['year_obs', 'country', 'city']).agg({
    'sim': 'mean',
    'obs': 'mean',
    'num_obs': 'sum',
    'lat': 'mean',
    'lon': 'mean' }).reset_index()
# Write results to Excel
with pd.ExcelWriter('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/' + 'Beijing_c360_sim_vs_SPARTAN_differeny_year.xlsx', engine='openpyxl') as writer:
    compr_df.to_excel(writer, sheet_name='Mon', index=False)
    annual_df.to_excel(writer, sheet_name='Annual', index=False)
################################################################################################
# Calculate monthly and annual average across different years
################################################################################################