import xarray as xr
import pandas as pd
import os
import time
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

years = range(2012, 2022)
data_list = []

#  START FULL EXECUTION TIMER
t0 = time.perf_counter()

for year in years:
    print(f"Started processing file: CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc")
    file_path = f"./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc"
    
    ds = xr.open_dataset(file_path)
    
    # Normalize coordinate names to lat/lon
    rename_map = {}
    if 'latitude' in ds.coords: rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)
    
    df = ds.to_dataframe().dropna().reset_index()
    
    # Add year column
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    
    data_list.append(df)
    
    ds.close()
    print(f"Finished processing file: {year}")

# Concatenate all data
data_sample = pd.concat(data_list, ignore_index=True)
print(len(data_sample))


# ── Aggregation per location ─────────────────────────────
agregation_per_location  = data_sample.groupby(['lat', 'lon'])['pr'].agg(
    avg_precipitation='mean',
    max_precipitation='max'
).reset_index()

agregation_per_location = agregation_per_location.round(3)
print(agregation_per_location.head())


# ── Yearly statistics ────────────────────────────────────
yearly_stats = data_sample.groupby('year')['pr'].agg(
    avg_pr='mean',
    max_pr='max'
).reset_index()

# Linear regression
slope, intercept, r, p, se = stats.linregress(yearly_stats['year'], yearly_stats['avg_pr'])
yearly_stats['trend'] = intercept + slope * yearly_stats['year']

print(yearly_stats)
print(f"Slope: {slope:.4f} mm/year | R²: {r**2:.4f} | p-value: {p:.4f}")


# ── Extreme events ───────────────────────────────────────
threshold_95 = data_sample['pr'].quantile(0.95)
extreme_events = data_sample[data_sample['pr'] > threshold_95]

print(f"95th percentile threshold : {threshold_95:.4f} mm")
print(f"Total extreme events      : {len(extreme_events):,}")
print(f"% of total records        : {100 * len(extreme_events) / len(data_sample):.2f}%")

extreme_per_year = extreme_events.groupby('year').size().reset_index(name='extreme_count')
print(extreme_per_year)


# ── Plot ────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
gs  = gridspec.GridSpec(3, 1, hspace=0.45)

years_vals = yearly_stats['year'].values

# Panel 1
ax1 = fig.add_subplot(gs[0])
ax1.plot(years_vals, yearly_stats['avg_pr'], marker='o', linewidth=2, markersize=6, label='Yearly average')
ax1.plot(years_vals, yearly_stats['trend'], linestyle='--', linewidth=2,
         label=f'Trend (slope={slope:.4f} mm/yr, R²={r**2:.3f})')
ax1.fill_between(years_vals, yearly_stats['avg_pr'], yearly_stats['trend'], alpha=0.08)
ax1.set_title('Yearly average precipitation with linear regression')
ax1.set_ylabel('Avg precipitation (mm)')
ax1.set_xticks(years_vals)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Panel 2
ax2 = fig.add_subplot(gs[1])
bars = ax2.bar(years_vals, yearly_stats['max_pr'])
for bar, val in zip(bars, yearly_stats['max_pr']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom')
ax2.set_title('Yearly maximum precipitation')
ax2.set_ylabel('Max precipitation (mm)')
ax2.set_xticks(years_vals)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# Panel 3
ax3 = fig.add_subplot(gs[2])
bars3 = ax3.bar(extreme_per_year['year'], extreme_per_year['extreme_count'])
for bar, val in zip(bars3, extreme_per_year['extreme_count']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:,}', ha='center', va='bottom')
ax3.axhline(extreme_per_year['extreme_count'].mean(),
            linestyle='--', linewidth=1.5,
            label=f"Mean: {extreme_per_year['extreme_count'].mean():,.0f}")
ax3.set_title(f'Extreme events per year (threshold: {threshold_95:.2f} mm)')
ax3.set_ylabel('Event count')
ax3.set_xticks(extreme_per_year['year'])
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.4)

plt.savefig('precipitation_analysis.png', dpi=150, bbox_inches='tight')
plt.show()


#  END FULL EXECUTION TIMER
t1 = time.perf_counter()

print(f"\n Total execution time (serial): {t1 - t0:.2f} seconds")