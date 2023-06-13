# scatter plot of RCFM and FM colored by site
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.font_manager as font_manager

# read the xlsx file
df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_output.xlsx', sheet_name='RCFM_Underestimated')

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex=True)

# set front and front size for legend and title
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

# plot RCFM and FM
plot1 = sns.scatterplot(data=df, x="FM", y="RCFM", hue="Site", ax=ax1, s=30, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.plot([df['FM'].min(), 200], [df['FM'].min(), 200], color='k', linestyle='--', alpha=0.8, linewidth=1)
ax1.set_title(label='Scatter plot of FM and RCFM', loc="center")
ax1.set_xlim([0, 160])
ax1.set_ylim([0, 160])
ax1.set_xticks([0, 50, 100, 150], fontname='Arial', size=14)
ax1.set_yticks([0, 50, 100, 150], fontname='Arial', size=14)
ax1.set_xlabel('FM (µg/m$^3$)', fontname='Arial', size=14)
ax1.set_ylabel('RCFM (µg/m$^3$)', fontname='Arial', size=14)
ax1.legend_.remove()

# plot RCFM_imposing and FM
plot2 = sns.scatterplot(data=df, x="FM", y="RCFM_imposing", hue="Site", ax=ax2, s=30, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.plot([df['FM'].min(), 200], [df['FM'].min(), 200], color='k', linestyle='--', alpha=0.8, linewidth=1)
ax2.set_title(label='Imposing OM/OC=2.5 and FM=1.5*RCFM threshold', loc="center")
ax2.set_xlim([0, 160])
ax2.set_ylim([0, 160])
ax2.set_xticks([0, 50, 100, 150], fontname='Arial', labelsize=14)
ax2.set_yticks([0, 50, 100, 150], fontname='Arial', size=14)
ax2.set_xlabel('FM (µg/m$^3$)', fontname='Arial', size=14)
ax2.set_ylabel('RCFM (µg/m$^3$)', fontname='Arial', size=14)

# create the legend
handles, labels = ax1.get_legend_handles_labels()
ax2.legend(handles, labels, loc='center right', bbox_to_anchor=(1.3, 0.5), ncol=1, title='Site')
fig.subplots_adjust(left=0, right=0.8, top=0.9, bottom=0.2, wspace=0)

# add the regression equation and r-squared
ax1.text(0.2, 0.9, "y = {:.2f}x + {:.2f}\nR$^2$ = {:.2f}".format(0.86, 6.41, 0.64),
         transform=ax1.transAxes, ha='center', va='center')
ax2.text(0.2, 0.9, "y = {:.2f}x + {:.2f}\nR$^2$ = {:.2f}".format(0.57, 8.95, 0.47),
         transform=ax2.transAxes, ha='center', va='center')

# export
plt.tight_layout()
plt.savefig("/Users/renyuxuan/Desktop/Research/RCFM_scatter_plot_FM=1.5RCFM.tiff", format="TIFF", dpi=300)
plt.show()
