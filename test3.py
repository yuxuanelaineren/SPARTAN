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

# Load the data from HIPS_SPARTAN.xlsx
obs_df = pd.read_excel(out_dir + 'HIPS_SPARTAN.xlsx')
CEDS_df = pd.read_excel(out_dir + 'C360_CEDS_noLUO_Sim_vs_SPARTAN_BC_2019_Summary.xlsx', sheet_name='Mon')

# obs_df = obs_df[obs_df['Site'] == 'CHTS']

# Group by 'start_year' and 'start_month', calculate average and count of 'BC_HIPS_ug'
compr_df = obs_df.groupby(['start_year', 'start_month', 'City'])['BC_HIPS_ug'].agg(['mean', 'count']).reset_index()

# Rename columns
compr_df.columns = ['year', 'mon', 'city', 'obs', 'num_obs']

# Save the result to a new Excel file
compr_df.to_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/Beijing_c360_sim_vs_SPARTAN_differeny_year.xlsx', index=False)
