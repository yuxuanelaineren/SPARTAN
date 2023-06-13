import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

########### create stacked bar plot with negative RM at th botton #############
# Load your data into a pandas DataFrame
df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_forPython.xlsx', sheet_name='BUUD_Outlier')
df["TEO"] = df["TEO"]*0.001

# Split the FilterID column into two parts
df["FilterID_1"] = df["FilterID"].str.split("-").str[0]
df["FilterID_2"] = "-" + df["FilterID"].str.split("-").str[1]

# Set up the plot
fig, ax = plt.subplots()

# set front and front size for legend and title
plt.rcParams.update({'font.family': 'Arial', 'font.size': 12})

# Set up the colors for each bar segment
colors = ['grey', 'red', 'blue', 'cyan', 'yellow', 'black', '#D55E00', 'green']
# Extract the negative values in the RM column
neg_values = df.loc[df["RM"] < 0, "RM"]

# Create the stacked bar plot
ax.bar(df["FilterID"], df["RM"], color=colors[0], label="RM")
ax.bar(df["FilterID"], df["AS"], bottom=df["RM"], color=colors[1], label="AS")
ax.bar(df["FilterID"], df["AN"], bottom=df["RM"] + df["AS"], color=colors[2], label="AN")
ax.bar(df["FilterID"], df["SS"], bottom=df["RM"] + df["AS"] + df["AN"], color=colors[3], label="SS")
ax.bar(df["FilterID"], df["Soil"], bottom=df["RM"] + df["AS"] + df["AN"] + df["SS"], color=colors[4], label="Soil")
ax.bar(df["FilterID"], df["EBC"], bottom=df["RM"] + df["AS"] + df["AN"] + df["SS"] + df["Soil"], color=colors[5], label="EBC")
ax.bar(df["FilterID"], df["TEO"], bottom=df["RM"] + df["AS"] + df["AN"] + df["SS"] + df["Soil"] + df["EBC"], color=colors[6], label="TEO")
ax.bar(df["FilterID"], df["OM"], bottom=df["RM"] + df["AS"] + df["AN"] + df["SS"] + df["Soil"] + df["EBC"] + df["TEO"], color=colors[7], label="OM")

# Set the negative values in the RM column to be facing down
ax.bar(df.loc[df["RM"] < 0, "FilterID"], neg_values, color=colors[0], label="RM", hatch="\\", edgecolor="white")

# Set the axis labels and legend
ax.set_ylabel('Concentration (µg/m$^3$)', fontname='Arial', size=14)
ax.set_title('Stacked bar plot for BDDU')
ax.legend(loc='upper right', ncol=2, title='Species')

# Rotate the x-tick labels by 90 degrees
plt.xticks(rotation=90, fontname='Arial')

# Set the x-tick labels to two rows
ax.set_xticklabels(df["FilterID_1"] + "\n" + df["FilterID_2"])

# Rotate the x-tick labels by 90 degrees
ax.set_yticks([0, 50, 100, 150])

# Show the plot
plt.tight_layout()
plt.savefig("/Users/renyuxuan/Desktop/Research/RCFM_plot_bar_BDDU.tiff", format="TIFF", dpi=300)
plt.show()



############### bar plot for summary the number
# Read the summary file
data_dir = '/Users/renyuxuan/Desktop/Research/RCFM'
path = os.path.join(data_dir, 'FTIR_OM_RM_all.xlsx')
df = pd.read_excel(path, sheet_name='Summary')

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))

# set front and front size for legend and title
plt.rcParams.update({'font.family': 'Arial', 'font.size': 14})

# Calculate the width of each bar
width = 0.25

# Create the first set of bars
ax.bar(df.index - width, df['# of OM'], width=width, color=sns.color_palette()[0], label='# of OM', edgecolor='black')

# Create the second set of bars
ax.bar(df.index, df['# of Residual'], width=width, color=sns.color_palette()[2], label='# of Residual', edgecolor='black')

# Create the third set of bars
ax.bar(df.index + width, df['matched # with positive Residual'], width=width, color=sns.color_palette()[3], label='matched # with positive Residual', edgecolor='black')

# Create the fourth set of bars
ax.bar(df.index + width, df['matched # with negative Residual'], bottom=df['matched # with positive Residual'], width=width, color='grey', label='matched # with negative Residual', edgecolor='black')

plt.title('Summary of the Number of Filters in FT-IR Organic Matter and Residual', fontname='Arial', fontsize=16, y=1.03)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticks(df.index)
ax.set_xticklabels(df['Site'], fontname='Arial', fontsize=12)
plt.yticks([0, 20, 40, 60, 80, 100, 120], fontname='Arial', fontsize=14)
legend=ax.legend(loc='upper right', ncol=2, fontsize=12)
legend.get_frame().set_linewidth(0.0)  # remove legend border

# Rotate the x-tick labels by 90 degrees
plt.xticks(rotation=45)

# show the plot
plt.tight_layout()
plt.savefig("/Users/renyuxuan/Desktop/Research/RCFM/OM_RM_Summary.tiff", format="TIFF", dpi=300)
plt.show()


