# scatter plot of RCFM and FM in all the sites with AEAZ and ILNZ colored
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as font_manager

# read the xlsx file to extract data
df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_forPython.xlsx', sheet_name='RCFM')
# 1:1 line
x = [0, 10, 100, 200, 300, 1000]
y = [0, 10, 100, 200, 300, 1000]
#sns.set_theme(style="darkgrid")
plt.figure(figsize=(8,6))
plt.plot(x, y, c='black', alpha=0.8, linewidth=1)
# scatter plot
sns.color_palette("tab10")
sns.scatterplot(x="RCFM", y="FM", data=df, hue="Site", s=30, alpha=0.8, edgecolor='black', linewidth=0.5)
plt.xlim([0, 200])
plt.ylim([0, 200])
plt.xticks(fontname='Arial', size=14)
plt.yticks(fontname='Arial', size=14)
plt.xlabel('Reconstructed PM$_{2.5}$ mass (µg/m$^3$)', fontname='Arial', size=14)
plt.ylabel('Gravimetric PM$_{2.5}$ mass (µg/m$^3$)', fontname='Arial', size=14)
plt.title(label='Scatter plot of FM and RCFM in the thirteen analyzed sites', loc="center", fontname='Arial', size=14)
# legend
font = font_manager.FontProperties(family='Arial', style='normal', size=14)
plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
plt.legend(loc='lower right', prop=font)
# export
plt.tight_layout()
plt.show()
plt.savefig("Scatter_plot.tiff", format='tiff', dpi=300)