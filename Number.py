# histogram of FT-IR OM number
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.font_manager as font_manager

# read the xlsx file to extract data
df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM_FT-IR_forPython.xlsx', sheet_name='Number')
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(df.Site.unique()))
bar_width = 0.4
b1 = ax.bar(x, df.loc[df['Type'] == 'Valid', 'Number'],
            width=bar_width)
b2 = ax.bar(x + bar_width, df.loc[df['Type'] == 'Valid after imposing', 'Number'],
            width=bar_width)
b3 = ax.bar(x + 0.8, df.loc[df['Type'] == 'Matched for RCFM', 'Number'],
            width=bar_width)
b4 = ax.bar(x + 1.4, df.loc[df['Type'] == 'Matched for RCFM after imposing', 'Number'],
            width=bar_width)


# export
plt.tight_layout()
plt.show()
plt.savefig("/Users/renyuxuan/Desktop/Research/Scatter_plot.tiff", format='tiff', dpi=300)