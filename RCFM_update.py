import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

##################### extract RM from RCFM file folder
# Set the directory path
dir_path = r'/Users/renyuxuan/Desktop/Research/RCFM/raw_data/RCFM'

# Create an empty dataframe to store the combined data
combined_data = pd.DataFrame()

# Create an empty dictionary to store the row count and negative value count by site name
row_count_by_site = {}
negative_count_by_site = {}

# Iterate over each file in the directory
for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):
        # Read the data from the file
        data = pd.read_csv(os.path.join(dir_path, filename), header=1)
        # Filter the rows with 'Residual Matter' in the 'Parameter_Name' column
        filtered_data = data[data['Parameter_Name'] == 'Residual Matter']
        # Extract the site name from the filename
        site_name = filename.split('_')[0]
        # Add the site name as a column in the filtered data
        filtered_data['Site'] = site_name
        # Append the filtered data to the combined data
        combined_data = combined_data.append(filtered_data, ignore_index=True)
        # Count the number of rows with 'Residual Matter'
        row_count = filtered_data.shape[0]
        # Count the number of negative values in the 'Value' column
        negative_count = (filtered_data['Value'] < 0).sum()
        # Add the row count and negative value count to the dictionaries
        if site_name in row_count_by_site:
            row_count_by_site[site_name] += row_count
        else:
            row_count_by_site[site_name] = row_count
        if site_name in negative_count_by_site:
            negative_count_by_site[site_name] += negative_count
        else:
            negative_count_by_site[site_name] = negative_count

# Create a dictionary to store the filtered dataframes by site name
filtered_data_by_site = {site: combined_data[combined_data['Site'] == site] for site in combined_data['Site'].unique()}

# Sort the dictionary by site name
filtered_data_by_site = dict(sorted(filtered_data_by_site.items()))

# Write the filtered dataframes to separate sheets in an Excel file
with pd.ExcelWriter('/Users/renyuxuan/Desktop/Research/RCFM/SPARTAN_Residual_Matter.xlsx') as writer:
    # Write the summary sheet
    summary_data = pd.DataFrame(row_count_by_site.items(), columns=['Site', 'Number'])
    # Add the negative value count column to the summary data
    summary_data['negative number'] = summary_data['Site'].map(negative_count_by_site)
    # Sort the data by site name
    summary_data = summary_data.sort_values(by='Site')
    # Write the summary data to a new sheet in the Excel file
    summary_data.to_excel(writer, sheet_name='Summary', index=False)
    # Write the filtered data to separate sheets
    for site, data in filtered_data_by_site.items():
        # Rename the columns to remove spaces
        data = data.rename(columns={col: col.replace(' ', '_') for col in data.columns})
        # Write the data to a new sheet with the site name
        data.to_excel(writer, sheet_name=site, index=False)

    # Combine all data into a new sheet named "All"
    all_data = pd.concat(filtered_data_by_site.values())
    all_data.to_excel(writer, sheet_name='All', index=False)

########################### match RM, OM and OC
# read RM file
# Read the RM file
data_dir = '/Users/renyuxuan/Desktop/Research/RCFM'
RM_path = os.path.join(data_dir, 'SPARTAN_Residual_Matter.xlsx')
RM_df = pd.read_excel(RM_path, sheet_name='All',
                      usecols=['Site', 'Start_Year_local', 'Start_Month_local', 'Start_Day_local', 'Value'])

# combine date columns into a single column
RM_df['Date'] = RM_df['Start_Year_local'].astype(str) + '_' + RM_df['Start_Month_local'].astype(str) + '_' + RM_df[
    'Start_Day_local'].astype(str)

# read OM file for 2020_09 sheet
OM_path = os.path.join(data_dir, 'FTIR_OM_RM_all.xlsx')
OM_2020_df = pd.read_excel(OM_path, sheet_name='2020_09', usecols=['Site', 'Date', 'OM', 'OC', 'Ratio'])

# convert date column to the desired format
OM_2020_df['Date'] = pd.to_datetime(OM_2020_df['Date']).dt.strftime('%Y_%-m_%-d')

# read OM file for 2022_06 sheet
OM_2022_df = pd.read_excel(OM_path, sheet_name='2022_06', usecols=['Site', 'Date', 'OM', 'OC', 'Ratio'])

# convert date column to the desired format
OM_2022_df['Date'] = pd.to_datetime(OM_2022_df['Date']).dt.strftime('%Y_%-m_%-d')

# merge RM and OM dataframes based on matching values of "Site" and "Date"
merged_df = pd.merge(RM_df, pd.concat([OM_2020_df, OM_2022_df]), on=['Site', 'Date'], how='inner')

# rename "Value" column to "RM"
merged_df.rename(columns={'Value': 'RM'}, inplace=True)

# write the merged dataframe to a new sheet in the OM file
with pd.ExcelWriter('/Users/renyuxuan/Desktop/Research/RCFM/FTIR_OM_RM_all.xlsx', engine='openpyxl',
                    mode='a') as writer:
    merged_df.to_excel(writer, sheet_name='OM_OC_RM', index=False)

###################################### count matched number
# read the OM file
OM_RM_df = pd.read_excel('/Users/renyuxuan/Desktop/Research/RCFM/FTIR_OM_RM_all.xlsx', sheet_name=None)

# concatenate all sheets into a single dataframe
OM_RM_df = pd.concat(OM_RM_df, ignore_index=True)

# calculate the number of rows and the number of negative values in RM for each site
summary_df = OM_RM_df.groupby(['Site']).agg({'RM': ['count', lambda x: (x < 0).sum()]})

# rename columns
summary_df.columns = ['Num_rows', 'Num_neg_RM']

# reset index
summary_df.reset_index(inplace=True)

# write summary dataframe to a new sheet
with pd.ExcelWriter('/Users/renyuxuan/Desktop/Research/RCFM/FTIR_OM_RM_all.xlsx', engine='openpyxl',
                    mode='a') as writer:
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

########## impose OM/OC = 2.5 threshold to OM _imposing and count rows with OM/OC > 2.5
# Read the OM_OC_RM file
data_dir = '/Users/renyuxuan/Desktop/Research/RCFM'
OM_RM_path = os.path.join(data_dir, 'FTIR_OM_RM_all.xlsx')
OM_RM_df = pd.read_excel(OM_RM_path, sheet_name='OM_OC_RM')

# add OM_imposing with 2.5 threshold
OM_RM_df['OM_imposing'] = OM_RM_df.apply(lambda row: row['OM'] if row['Ratio'] < 2.5 else row['OC']*2.5, axis=1)

# write dataframe to a new sheet
with pd.ExcelWriter('/Users/renyuxuan/Desktop/Research/RCFM/FTIR_OM_RM_all.xlsx', engine='openpyxl',
                    mode='a') as writer:
    OM_RM_df.to_excel(writer, sheet_name='OM_OC_RM_imposing', index=False)

# Read the OM_OC_RM file
data_dir = '/Users/renyuxuan/Desktop/Research/RCFM'
OM_RM_path = os.path.join(data_dir, 'FTIR_OM_RM_all.xlsx')
OM_RM_df = pd.read_excel(OM_RM_path, sheet_name='OM_OC_RM_imposing')

# Count number of rows where Ratio < 1 and Ratio > 2.5 for each site
ratio_counts = OM_RM_df.groupby(['Site', pd.cut(OM_RM_df['Ratio'], [-np.inf, 1, 2.5, np.inf])])['Ratio'].size().unstack(fill_value=0)

# Rename columns to be more descriptive
ratio_counts.columns = ['Ratio < 1', '1 <= Ratio <= 2.5', 'Ratio > 2.5']

# Write summary dataframe to a new sheet
with pd.ExcelWriter('/Users/renyuxuan/Desktop/Research/RCFM/FTIR_OM_RM_all.xlsx', engine='openpyxl', mode='a') as writer:
    ratio_counts.to_excel(writer, sheet_name='Summary_Ratio')

####################### create scatter plot for all sites
# Read the OM file
data_dir = '/Users/renyuxuan/Desktop/Research/RCFM'
OM_RM_path = os.path.join(data_dir, 'FTIR_OM_RM_all.xlsx')
OM_RM_df = pd.read_excel(OM_RM_path, sheet_name='OM_OC_RM')

# Drop rows where RM is greater than 200
OM_RM_df = OM_RM_df.loc[OM_RM_df['RM'] <= 200]

# Drop rows where RM is negative
OM_RM_positive_df = OM_RM_df.loc[OM_RM_df['RM'] >= 0]

# Create figure and axes objects
fig, ax = plt.subplots(figsize=(8, 6))

# Create scatter plot with white background, black border, and no grid
sns.set(font='Arial')
scatterplot = sns.scatterplot(x='OM', y='RM', data=OM_RM_positive_df, hue='Site', s=15, alpha=1, ax=ax)
scatterplot.set_facecolor('white')  # set background color to white
border_width = 1  # specify the width of the border
for spine in scatterplot.spines.values():
    spine.set_edgecolor('black')  # set border color to black
    spine.set_linewidth(border_width)  # set border width
scatterplot.grid(False)  # remove the grid

# Modify legend background color and position
legend = plt.legend(facecolor='white', bbox_to_anchor=(1.03, 0.5), loc='center left', fontsize=12)
# legend.get_frame().set_linewidth(0.0)  # remove legend border
legend.get_texts()[0].set_fontname("Arial")  # set fontname of the first label

# Set title, xlim, ylim, ticks, labels
plt.title('Comparison of FT-IR Organic Matter and Residual', fontname='Arial', fontsize=16, y=1.03)
plt.xlim([OM_RM_positive_df['RM'].min()-0.4, OM_RM_positive_df['RM'].max()+0.4])
plt.ylim([OM_RM_positive_df['RM'].min()-0.4, OM_RM_positive_df['RM'].max()+0.4])
plt.xticks([0, 10, 20, 30], fontname='Arial', size=14)
plt.yticks([0, 10, 20, 30], fontname='Arial', size=14)
scatterplot.tick_params(axis='x', direction='out', width=1, length=5)
scatterplot.tick_params(axis='y', direction='out', width=1, length=5)

# Add 1:1 line with black dash
x = OM_RM_positive_df['RM']
y = OM_RM_positive_df['RM']
plt.plot([OM_RM_positive_df['RM'].min(), 45], [OM_RM_positive_df['RM'].min(), 45], color='grey', linestyle='--', linewidth=1)

# Add linear regression, function, and r2 to the plot
sns.regplot(x='OM', y='RM', data=OM_RM_positive_df, scatter=False, ci=None, line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1})
slope, intercept, r_value, p_value, std_err = stats.linregress(OM_RM_positive_df['OM'], OM_RM_positive_df['RM'])
plt.text(0.05, 0.85, f"y = {slope:.2f}x + {intercept:.2f}\n$r^2$ = {r_value ** 2:.2f}",
         transform=scatterplot.transAxes, fontsize=14)

# Add number of data points to the plot
num_points = len(OM_RM_positive_df)
plt.text(0.05, 0.8, f'N = {num_points}', transform=scatterplot.transAxes, fontsize=14)

# Set x-axis and y-axis labels
plt.xlabel('FT-IR Organic Matter (µg/m$^3$)', fontsize=14, color='black')
plt.ylabel('Residual (µg/m$^3$)', fontsize=14, color='black')

# show the plot
plt.tight_layout()
plt.savefig("/Users/renyuxuan/Desktop/Research/RCFM/OM_RM_positive.tiff", format="TIFF", dpi=300)
plt.show()

#################### create scatter plot for each site as subplot
# Define variables
DATA_DIR = '/Users/renyuxuan/Desktop/Research/RCFM'
OM_RM_PATH = os.path.join(DATA_DIR, 'FTIR_OM_RM_all.xlsx')
NUM_COLS = 4
FIG_SIZE = (12, 12)

# Read the OM file
try:
    om_rm_df = pd.read_excel(OM_RM_PATH, sheet_name='OM_OC_RM')
except FileNotFoundError:
    print(f"{OM_RM_PATH} not found.")
    exit()

# Filter out invalid data
om_rm_df = om_rm_df.loc[(om_rm_df['RM'] >= 0) & (om_rm_df['RM'] <= 200)]

# Get a list of all unique site names
site_names = om_rm_df['Site'].unique()

# Count the number of non-empty plots
num_plots = 0
for site_name in site_names:
    site_data = om_rm_df.loc[om_rm_df['Site'] == site_name]
    if len(site_data) >= 2:
        num_plots += 1

# Adjust number of rows and columns based on number of non-empty plots
NUM_COLS = 4
NUM_ROWS = int(np.ceil(num_plots / NUM_COLS))

# Create a figure with subplots for each site
fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=FIG_SIZE, sharex=True, sharey=True)

# Create scatter plots for each site
sns.set(font='Arial')
plot_idx = 0
for site_name in site_names:

    # Get the data for the current site
    site_data = om_rm_df.loc[om_rm_df['Site'] == site_name]

    # Skip site if no data points
    if len(site_data) < 2:
        continue

    # Determine the row and column indices for the current plot
    row_idx = plot_idx // NUM_COLS
    col_idx = plot_idx % NUM_COLS

    # Create subplot for site
    ax = axs[row_idx, col_idx]

    # Create scatter plot with white background, black border, and no grid
    scatterplot = sns.scatterplot(x='OM', y='RM', data=site_data, s=15, alpha=1, ax=ax)
    scatterplot.set_facecolor('white')  # set background color to white
    border_width = 1  # specify the width of the border
    for spine in scatterplot.spines.values():
        spine.set_edgecolor('black')  # set border color to black
        spine.set_linewidth(border_width)  # set border width
    scatterplot.grid(False)  # remove the grid

    # Show x and y ticks for the current subplot
    scatterplot.tick_params(axis='both', which='both', labelsize=10, length=4)
    scatterplot.xaxis.set_tick_params(labelbottom=True)
    scatterplot.yaxis.set_tick_params(labelleft=True)

    # Add 1:1 line with grey dash
    x = site_data['RM']
    y = site_data['RM']
    ax.plot([om_rm_df['RM'].min(), om_rm_df['OM'].max()], [om_rm_df['RM'].min(), om_rm_df['OM'].max()],
            color='grey', linestyle='--', linewidth=1)

    # Add linear regression, function, and r2 to the plot
    sns.regplot(x='OM', y='RM', data=site_data, scatter=False, ci=None,
                line_kws={'color': 'k', 'linestyle': '-', 'linewidth': 1}, ax=ax)
    slope, intercept, r_value, p_value, std_err = stats.linregress(site_data['OM'], site_data['RM'])
    ax.text(0.05, 0.7, f"y = {slope:.2f}x + {intercept:.2f}\n$r^2$ = {r_value ** 2:.2f}",
            transform=ax.transAxes, fontsize=12)

    # Add number of data points to the plot
    num_points = len(site_data)
    ax.text(0.05, 0.58, f'N = {num_points}', transform=ax.transAxes, fontsize=12)

    # Set plot title as site name
    ax.set_title(site_name, fontname='Arial', fontsize=12, y=1.03)

    plot_idx += 1

# Loop through subplots and remove x and y labels
for ax in axs.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

# set x and y labels above the subplots
fig.text(0.5, 0.05, 'FT-IR Organic Matter (µg/m$^3$)', ha='center', va='center', fontsize=14)
fig.text(0.05, 0.5, 'Residual (µg/m$^3$)', ha='center', va='center', rotation='vertical', fontsize=14)

# Adjust vertical distance among subplots
fig.subplots_adjust(hspace=0.4)
fig.tight_layout()
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.1)

plt.savefig("/Users/renyuxuan/Desktop/Research/RCFM/OM_RM_positive_each_site.tiff", format="TIFF", dpi=300)
plt.show()











