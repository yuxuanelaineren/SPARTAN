import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
# Load the data from the Excel file
obs_df = pd.read_excel(
    '/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_HIPS_FTIR_UV-Vis_SPARTAN_allYears.xlsx',
    sheet_name='Summary'
)

# Filter out cities where all counts are 0
obs_df = obs_df[(obs_df['count_BC_HIPS'] != 0) |
                (obs_df['count_EC_FTIR'] != 0) |
                (obs_df['count_BC_UV_Vis'] != 0)]

# Melt the DataFrame for easier plotting
melted_df = obs_df.melt(
    id_vars=['Country', 'City',
             'earliest_date_HIPS', 'latest_date_HIPS',
             'earliest_date_EC_FTIR', 'latest_date_EC_FTIR',
             'earliest_date_UV_Vis', 'latest_date_UV_Vis'],
    value_vars=['count_BC_HIPS', 'count_EC_FTIR', 'count_BC_UV_Vis'],
    var_name='Measurement',
    value_name='Count'
)
# Create a custom color palette
custom_palette = {
    'count_BC_HIPS': 'green',  # Black for HIPS
    'count_EC_FTIR': 'blue',    # Blue for FT-IR
    'count_BC_UV_Vis': 'red'     # Red for UV-Vis
}
# Create bar plot
plt.figure(figsize=(12, 8))
bar_plot = sns.barplot(data=melted_df, x='City', y='Count', hue='Measurement',
                        hue_order=['count_BC_HIPS', 'count_EC_FTIR', 'count_BC_UV_Vis'],
                        palette=custom_palette)




# Rotate x-ticks for better readability
plt.xticks(rotation=90)
plt.xlabel('City')
plt.ylabel('Count')
plt.title('Counts of HIPS, FT-IR, and UV-Vis Measurements')
plt.ylim(0, 320)
# Create a dictionary to map each measurement to its earliest and latest dates
measurements_dict = {
    'count_BC_HIPS': ('earliest_date_HIPS', 'latest_date_HIPS', 'HIPS'),
    'count_EC_FTIR': ('earliest_date_EC_FTIR', 'latest_date_EC_FTIR', 'EC FTIR'),
    'count_BC_UV_Vis': ('earliest_date_UV_Vis', 'latest_date_UV_Vis', 'UV Vis')
}

# Add earliest and latest date as annotations for each measurement
for index, row in melted_df.iterrows():
    city = row['City']
    measurement = row['Measurement']

    # Get earliest and latest dates from the dictionary
    earliest_date_col, latest_date_col, label = measurements_dict[measurement]
    earliest_date = obs_df.loc[obs_df['City'] == city, earliest_date_col].values[0]
    latest_date = obs_df.loc[obs_df['City'] == city, latest_date_col].values[0]

    # Format the dates as strings without measurement labels
    def format_date(date):
        if pd.isna(date):  # Check if the date is NaT
            return ''
        else:
            return pd.to_datetime(date).strftime('%Y-%m-%d')

    # Construct date ranges for annotations
    earliest_date_str = format_date(earliest_date)
    latest_date_str = format_date(latest_date)
    date_range = f"{earliest_date_str} to {latest_date_str}"
    # Check if both dates are valid and construct date range accordingly
    date_range = ""
    if earliest_date_str and latest_date_str:  # Both dates are valid
        date_range = f"{earliest_date_str} to {latest_date_str}"
    elif earliest_date_str:  # Only earliest date is valid
        date_range = earliest_date_str
    elif latest_date_str:  # Only latest date is valid
        date_range = latest_date_str

    # Calculate bar position: add 0.2 for spacing between groups
    bar_pos = index % len(obs_df['City'].unique()) + (index // len(obs_df['City'].unique())) * 0.3
    # Adding the annotation above the bar
    plt.text(bar_pos - 0.25, row['Count'] + 2,  # Adjust gap as needed
             date_range,
             ha='center', fontsize=8, color='k', fontweight='regular', rotation=90)
handles, labels = bar_plot.get_legend_handles_labels()
custom_labels = ['HIPS', 'FT-IR', 'UV-Vis']
bar_plot.legend(handles, custom_labels, loc='upper left', frameon=True)  # frameon=False removes the outer line

plt.tight_layout()
plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/BC_counts.tiff', dpi=300)
plt.show()