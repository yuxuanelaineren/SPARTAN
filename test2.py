import matplotlib.pyplot as plt
import numpy as np

# Data for Dhaka and Halifax
species = ['Soil', 'Al', 'Si', 'Ca', 'Fe', 'Ti', 'TEO', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Rb', 'Sr', 'Cd', 'Sn', 'Sb', 'Ce', 'Pb']
precision_dhaka = [0.39, 0.35, 0.45, 0.22, 0.36, 0.40, 0.40, 0.51, 0.90, 0.47, 0.57, 2.04, 1.51, 0.46, 0.47, 0.68, 0.44, 0.52, 1.42, 32.95, 0.61, 0.44, 0.55]
precision_halifax = [0.17, 0.19, 0.18, 0.50, 0.22, 1.61, 1.67, 0.20, 0.49, 0.26, 1.48, 1.26, 48.48, 0.36, 0.14, 0.20, 3.31, 1.52, 2.54, 2.26, 1.53, 0.36, 0.46]

# Modify the precision values for 'Cu' and 'Sn' by multiplying by 0.1
cu_index = species.index('Cu')
Sn_index = species.index('Sn')
precision_dhaka[cu_index] *= 0.1
precision_halifax[cu_index] *= 0.1
precision_dhaka[Sn_index] *= 0.1
precision_halifax[Sn_index] *= 0.1

# Set up the bar width and positions
bar_width = 0.35
index = np.arange(len(species))

# Create the plot
fig, ax = plt.subplots(figsize=(8, 9))

# Plotting bars for Dhaka and Halifax
bar1 = ax.barh(index, precision_dhaka, bar_width, label='Dhaka', color='b')
bar2 = ax.barh(index + bar_width, precision_halifax, bar_width, label='Halifax', color='r')
border_width = 1
for spine in ax.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
ax.grid(False)  # remove the grid
# Adjust the species label for 'Cu' and 'Sn' to reflect the adjusted precision
species[cu_index] = 'Cu × 0.1'
species[Sn_index] = 'Sn × 0.1'

# Add a grey dashed line before "TEO"
teo_index = species.index('TEO')
ax.axhline(y=teo_index - bar_width / 2, color='grey', linestyle='--', linewidth=1)

# Adding labels, title, and ticks
ax.set_ylabel('Species', fontsize=16, family='Arial')
ax.set_xlabel('Precision', fontsize=16, family='Arial')
ax.set_title('Precision for Dhaka and Halifax', fontsize=16, family='Arial')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(species, fontsize=16, family='Arial')
plt.xlim([0, 5])
plt.xticks([0, 1, 2, 3, 4, 5], fontname='Arial', size=18)
ax.tick_params(axis='x', direction='out', width=1, length=5)
ax.tick_params(axis='y', direction='out', width=1, length=5)
# Reverse the y-axis order
ax.invert_yaxis()

# Set the legend with correct font properties
legend=plt.legend(fontsize=18, prop={'family': 'Arial'})
legend.get_frame().set_edgecolor('white')

# Display the plot
plt.tight_layout(pad=2)
Colocation_dir = '/Volumes/rvmartin/Active/ren.yuxuan/Mass_Reconstruction/Co-location/'
plt.savefig(Colocation_dir + 'Co-location_Precision_Dust_TEO.svg', dpi=300)

plt.show()
