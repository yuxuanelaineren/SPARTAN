################################################################################################
# Beijing: Plot seasonal variations
################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Read the data
compr_df = pd.read_excel('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/Beijing_c360_sim_vs_SPARTAN_differeny_year.xlsx', sheet_name='Mon')

compr_df = compr_df[compr_df['city'] == 'Beijing']

# Calculate the difference
compr_df['diff'] = compr_df['obs'] - compr_df['sim']


# Print column names
print(compr_df.columns)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot data for each year with different marker styles and colors
for year, data in compr_df.groupby('year_obs'):
    if year == 2020:
        ax.plot(data['month'], data['diff'], markersize=8, marker='o', linestyle='None', label=str(int(year)),
                color='red', markeredgewidth=0.5, markeredgecolor='black')
    elif year == 2022:
        ax.plot(data['month'], data['diff'], markersize=8, marker='s', linestyle='None', label=str(int(year)),
                color='blue', markeredgewidth=0.5, markeredgecolor='black')
    elif year == 2023:
        ax.plot(data['month'], data['diff'], markersize=8, marker='^', linestyle='None', label=str(int(year)),
                color='green', markeredgewidth=0.5, markeredgecolor='black')


border_width = 1
plt.ylim([-10, 25])
plt.xticks([0, 2, 4, 6, 8, 10, 12], fontname='Arial', size=18)
plt.yticks([-10, -5, 0, 5, 10, 15, 20, 25], fontname='Arial', size=18)


# Add y = 0 with grey dash
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)

# Add labels and title
plt.title('Seasonal variations in BC Difference in Beijing', fontsize=16, fontname='Arial', y=1.03)
plt.xlabel('Month', fontsize=18, color='black', fontname='Arial')
plt.ylabel('Difference (Observation - Simulation) (µg/m$^3$)', fontsize=18, color='black', fontname='Arial')

# Customize legend
legend = plt.legend(fontsize=16, frameon=True)
legend.get_frame().set_edgecolor('black')  # Set legend border color
legend.get_frame().set_linewidth(border_width)  # Set legend border width

# Set legend font to Arial
prop = fm.FontProperties(family='Arial', size=16)
plt.setp(legend.get_texts(), fontproperties=prop)

# Calculate average and count for each year
average_diff = compr_df.groupby('year_obs')['diff'].mean()
count_diff = compr_df.groupby('year_obs')['diff'].count()

# Add text in the bottom left corner
text = 'Yearly Average:\n'
for year, avg_diff in average_diff.items():
    text += f'{int(year)}: {avg_diff:.2f} µg/m$^3$ (n={count_diff[year]})\n'

plt.text(0.05, 0.0000000000000001, text, transform=ax.transAxes, fontsize=16, fontname='Arial', horizontalalignment='left')

plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/representative_bias/Beijing_c360_CEDS_BC_MonMean.tiff', dpi=600)

# Show the plot
plt.show()