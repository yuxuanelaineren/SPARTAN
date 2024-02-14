import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Load the data from Excel
file_path = "/Users/renyuxuan/Downloads/BC_Emissions.xlsx"
data = pd.read_excel(file_path)

# Load world map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge BC data with world map data based on country names
world = world.merge(data, how='left', left_on='name', right_on='Country')

# Plot the world map with BC values (setting color scheme and country boundaries)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# Fill missing country data with grey
plot = world.plot(column='BC', cmap='RdYlBu_r', linewidth=1, ax=ax, edgecolor='grey', legend=True,
                  legend_kwds={'label': "Black Carbon Emissions (Gg/year)",
                               'orientation': "vertical",
                               },
                  vmin=0, vmax=25, missing_kwds={'color': 'lightgrey'})

plot.set_title('BC emissions from open waste burning', fontname='Arial', fontsize=18)  # Set title font and size

# Get legend object
legend = ax.get_legend()

# Add text to the bottom-left corner
# plt.text(0, 0, '(Wiedinmyer et al., 2014)', fontsize=14, color='black', ha='left', va='bottom', transform=ax.transAxes)

# Turn off ticks and tick labels
plot.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
plot.set_xticklabels([])
plot.set_yticklabels([])

# Set all labels to Arial
for label in (plot.get_xticklabels() + plot.get_yticklabels()):
    label.set_fontname('Arial')


plt.savefig('/Users/renyuxuan/Downloads/BC_Emissions.tiff', dpi=600)

plt.show()


