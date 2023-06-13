import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the data from the Excel file
data = pd.read_excel('/Users/renyuxuan/Library/CloudStorage/OneDrive-WashingtonUniversityinSt.Louis/IC_LQL_Data/Figure_IC_LQL_Example.xlsx')

# Extract the SO4_all column
so4_all = data['SO4_all']
so4_95 = data['SO4_95']
so4_90 = data['SO4_90']

# set front and front size for legend and title
plt.rcParams.update({'font.family': 'Arial', 'font.size': 14})

# Create the histogram with 10 bins and figsize=(10, 6)
n_bins = 100
fig, ax = plt.subplots(figsize=(8, 6))
n, bins, patches = ax.hist(so4_all, bins=n_bins, color='grey', edgecolor='black')

# Set the axis labels and title with Arial font
ax.set_xlabel('Mass (μg)', fontname='Arial', size=14)
ax.set_ylabel('Number', fontname='Arial', size=14)
ax.set_title('The distribution of sulfate mass of the field blank filters', fontname='Arial', size=14, y=1.03)
ax.set_ylim([0, 70])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70], fontname='Arial', size=14)
ax.set_xlim([0, 120])
ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], fontname='Arial', size=14)

# Set x-axis tick locator to Auto
# ax.xaxis.set_major_locator(ticker.AutoLocator())

# Add major grid lines
ax.grid(which='both', linestyle='--', alpha=0.5)

# add the number
ax.text(110, 60, "N = 81", ha='center', va='center')

# show the plot
plt.tight_layout()
plt.savefig("/Users/renyuxuan/Library/CloudStorage/OneDrive-WashingtonUniversityinSt.Louis/IC_LQL_Data/IC_LQL_SO4.tiff", format="TIFF", dpi=300)
plt.show()


