import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager as fm
################################################################################################
# Beijing: Plot seasonal variations
################################################################################################
# Read the data
df = pd.read_csv('/Volumes/rvmartin/Active/SPARTAN-shared/Analysis_Data/Master_files/CHTS_master.csv')
# Define the columns to keep
BC_columns = ['FilterID', 'start_year', 'start_month', 'start_day', 'Mass_type', 'mass_ug', 'Volume_m3', 'BC_SSR_ug',
                'BC_HIPS_ug', 'Flags']
df = df[BC_columns].copy()

# Convert columns to numeric, drop rows with NaN
df['Mass_type'] = pd.to_numeric(df['Mass_type'], errors='coerce')
df = df.loc[df['Mass_type'] == 1]
numeric_columns = ['start_year', 'start_month', 'start_day', 'Volume_m3', 'BC_SSR_ug', 'BC_HIPS_ug']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
# Create a datetime column by combining year, month, and day
df['start_year'] = pd.to_numeric(df['start_year'], errors='coerce').fillna(0).astype(int)
df['start_month'] = pd.to_numeric(df['start_month'], errors='coerce').fillna(0).astype(int)
df['start_day'] = pd.to_numeric(df['start_day'], errors='coerce').fillna(0).astype(int)
df = df[(df['start_month'] >= 1) & (df['start_month'] <= 12) &
        (df['start_day'] >= 1) & (df['start_day'] <= 31)]
# Combine year, month, and day into a single datetime column
df['datetime'] = pd.to_datetime(df['start_year'].astype(str) + '-' +
                                df['start_month'].astype(str).str.zfill(2) + '-' +
                                df['start_day'].astype(str).str.zfill(2))
# Sort data by datetime
df = df.sort_values('datetime')
df = df.dropna(subset=['start_year', 'start_month', 'start_day', 'Volume_m3'])
df = df[df['Volume_m3'] > 0]
df_SSR = df[df['BC_SSR_ug'] > 0]
df_SSR = df.dropna(subset=['BC_SSR_ug'])
df_HIPS = df[df['BC_HIPS_ug'] > 0]
df_HIPS = df.dropna(subset=['BC_HIPS_ug'])

# Calculate BC concentration
df_SSR['BC_SSR'] = df_SSR['BC_SSR_ug'] / df_SSR['Volume_m3']
df_SSR = df_SSR[df_SSR['BC_SSR'] > 0.5]
df_HIPS['BC_HIPS'] = df_HIPS['BC_HIPS_ug'] / df_HIPS['Volume_m3']

# Plot the time series
plt.figure(figsize=(12, 6))
# plt.plot(df['datetime'], df['BC_conc'], marker='o', linestyle='None', markersize=8, markeredgewidth=0.5, markeredgecolor='black')

# Plot BC_SSR with blue dots and BC_HIPS with red triangles
plt.plot(df_SSR['datetime'], df_SSR['BC_SSR'], marker='o', linestyle='None', color='#4682B4', markersize=8, markeredgewidth=0.5, markeredgecolor='black', label='SSR')
plt.plot(df_HIPS['datetime'], df_HIPS['BC_HIPS'], marker='^', linestyle='None', color='#FF6347', markersize=8, markeredgewidth=0.5, markeredgecolor='black', label='HIPS')

# Add legend
legend = plt.legend(['SSR', 'HIPS'], loc='best', frameon=True, fancybox=True, framealpha=1, borderpad=0.5)
legend.get_frame().set_edgecolor('black')
prop = fm.FontProperties(family='Arial', size=16)
plt.setp(legend.get_texts(), fontproperties=prop)

# Formatting the plot
border_width = 1
# Use MonthLocator or YearLocator for fewer ticks
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(fontname='Arial', size=12, rotation=45)
plt.ylim([0, 20])
plt.yticks([0, 3, 6, 9, 12, 15, 18], fontname='Arial', size=18)
plt.xlabel('Date', fontname='Arial', size=18)
plt.ylabel('BC Concentration (µg/m$^3$)', fontsize=18, fontname='Arial')
plt.title('Time Series of BC Concentration in Beijing', fontsize=20, color='black', fontname='Arial')
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tight_layout()
plt.savefig('/Volumes/rvmartin/Active/ren.yuxuan/BC_Comparison/SPARTAN_BC/' + 'BC_TimeSeries_Beijing_SSR_HIPS.svg', dpi=300)
plt.show()