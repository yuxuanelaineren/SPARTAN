import matplotlib.pyplot as plt
from matplotlib import font_manager

# Emission values and labels
emission_values = [
    6.285354181567726e-13,  # BC_ene
    3.2386090337288564e-11,  # BC_ind
    4.7714308198942845e-11,  # BC_rco
    3.3088639467271364e-11,  # BC_tra
    8.401655239576034e-12  # BC_wst
]

labels = [
    'Energy',
    'Industrial',
    'Residential, Commercial, Other Combustion',
    'Transportation',
    'Waste'
]

# Colors for the pie chart
colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99', '#c2c2f0']

# Font properties
font_properties = font_manager.FontProperties(family='Arial', size=14)

# Create pie chart
fig, ax = plt.subplots(figsize=(11, 5))
wedges, texts, autotexts = ax.pie(emission_values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
                                  wedgeprops=dict(edgecolor='black', linewidth=1, width=1))

# Customize font properties for labels and percentages
for text in texts:
    text.set_fontproperties(font_properties)
for autotext in autotexts:
    autotext.set_fontproperties(font_properties)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')

plt.title('CEDS0.1 BC Emissions at Beijing Site', fontproperties=font_properties, y=1.05)
plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/c360_CEDS_noLUO_2019/FigSX_Beijing_CEDS0.1v2024-06_BC.tiff', dpi=600)

plt.show()
