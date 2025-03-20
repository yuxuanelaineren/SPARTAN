import pandas as pd

# Load the data from the first Excel file
fe_bc_path = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/Fe_BC_HIPS_SPARTAN.xlsx'
fe_bc_df = pd.read_excel(fe_bc_path, sheet_name='All')

# Load the data from the second Excel file
bc_after_screening_path = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_SPARTAN_afterScreening.xlsx'
bc_after_screening_df = pd.read_excel(bc_after_screening_path, sheet_name='All')

# Merge the dataframes based on 'FilterID' column and keep all columns from bc_after_screening_df
merged_df = pd.merge(bc_after_screening_df,
                     fe_bc_df[['FilterID', 'Fe']],
                     on='FilterID',
                     how='left')

# Write the merged dataframe to a new Excel file
output_path = '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/Fe_BC_HIPS_SPARTAN_afterScreening.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    merged_df.to_excel(writer, sheet_name='All', index=False)

print("Merge completed and saved to new Excel file.")
