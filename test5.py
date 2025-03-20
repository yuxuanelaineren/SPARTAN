import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the annual data from the Excel sheet
obs_annual_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/Fe_BC_HIPS_SPARTAN_afterScreening.xlsx', sheet_name='Annual')

# Multiply annual BC by 0.1
obs_annual_df['annual_mean_BC_adjusted'] = obs_annual_df['annual_mean_BC'] * 0.1
obs_annual_df['annual_mean_BC_corrected_adjusted'] = obs_annual_df['annual_mean_BC_corrected'] * 0.1
# Set up the plot style using seaborn
sns.set(style="whitegrid")

# Create the figure
fig, ax = plt.subplots(figsize=(12, 6))

# Set positions for the bars (x-axis positions)
x_pos = np.arange(len(obs_annual_df))

# Define bar width
bar_width = 0.2

# Plot all four bars side by side for each city
ax.bar(x_pos - 1.5*bar_width, obs_annual_df['annual_mean_BC_adjusted'], width=bar_width, label='BC × 0.1', color='b')
ax.bar(x_pos - 0.5*bar_width, obs_annual_df['annual_mean_BC_corrected_adjusted'],  label='corrected BC × 0.1', width=bar_width, color='g')
ax.bar(x_pos + 0.5*bar_width, obs_annual_df['annual_mean_Fe'], width=bar_width, label='Fe', color='r')
# Customize the plot
# plt.ylim([0, 1])
ax.set_xlabel('City', fontsize=14)
ax.set_ylabel('Concentration (µg/m$^3$)', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(obs_annual_df['City'], rotation=45, ha='right')
# ax.set_title('BC and Fe by City', fontsize=16)

# Add legend
ax.legend()

# Show the plot
plt.tight_layout()

plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/correctedBC_HIPS_SPARTAN.svg', dpi=300)
plt.show()