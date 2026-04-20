import time
import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.dataframe as dd
from dask.distributed import Client
import matplotlib.pyplot as plt
from scipy import stats


# ──────────────────────────────────────────────
# STEP 1 — OPEN ALL FILES AS ONE LAZY DATASET
# ──────────────────────────────────────────────
file_pattern = "./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_*_v2.0.nc"

print("Opening all files lazily with Dask...")
t0 = time.perf_counter()

def normalize_coords(ds):
    """Rename latitude/longitude to lat/lon if needed."""
    rename_map = {}
    if 'latitude'  in ds.coords: rename_map['latitude']  = 'lat'
    if 'longitude' in ds.coords: rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)
    return ds

ds = xr.open_mfdataset(
    file_pattern,
    combine="nested",
    concat_dim="time",
    chunks={"time": 365},
    parallel=True,
    preprocess=normalize_coords,
)

print(f"Dataset opened in {time.perf_counter() - t0:.2f}s  (lazy — no data read yet)")
print(ds)


# ──────────────────────────────────────────────
# STEP 2 — LOAD THE FULL DATASET YEAR BY YEAR
# ──────────────────────────────────────────────
# We still process one year at a time to keep peak memory manageable,
# but every row is kept — no sampling.

print("\nLoading full dataset year by year...")
t1 = time.perf_counter()

yearly_files = sorted(
    [f"./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_{y}_v2.0.nc"
     for y in range(2012, 2022)]
)

chunks = []
for f in yearly_files:
    year_ds = xr.open_dataset(f)
    year_ds = normalize_coords(year_ds)

    year_df = year_ds['pr'].to_dataframe().dropna().reset_index()
    year_df['time'] = pd.to_datetime(year_df['time'])
    year_df['year'] = year_df['time'].dt.year

    # ── NO sampling — keep every row ──
    chunks.append(year_df)

    year_ds.close()
    print(f"  loaded {year_df['year'].iloc[0]}  ({len(year_df):,} rows)")

df = pd.concat(chunks, ignore_index=True)

print(f"\nData loaded in {time.perf_counter() - t1:.2f}s")
print(f"Total rows: {len(df):,}")


# ──────────────────────────────────────────────
# STEP 3 — PARALLEL GROUPBY WITH DASK DATAFRAME
# ──────────────────────────────────────────────
# With the full dataset the DataFrame will be much larger, so we use more
# partitions to keep each partition a reasonable size for Dask workers.

print("\nRunning parallel groupby aggregations with Dask...")
t2 = time.perf_counter()

# Scale partitions with data size: ~1 partition per ~5 M rows, minimum 10
n_partitions = max(10, len(df) // 5_000_000)
ddf2 = dd.from_pandas(df, npartitions=n_partitions)
print(f"  Dask DataFrame: {n_partitions} partitions, {len(df):,} total rows")

# ── Analysis 1: Spatial aggregation ──
spatial_agg = (
    ddf2
    .groupby(['lat', 'lon'])['pr']
    .agg(['mean', 'max'])
    .rename(columns={'mean': 'avg_precipitation', 'max': 'max_precipitation'})
    .reset_index()
    .compute()
)

# ── Analysis 2: Temporal trend ──
yearly_agg = (
    ddf2
    .groupby('year')['pr']
    .agg(['mean', 'max'])
    .rename(columns={'mean': 'avg_precipitation', 'max': 'max_precipitation'})
    .reset_index()
    .compute()
    .sort_values('year')
)

slope_avg, intercept_avg, r_avg, p_avg, _ = stats.linregress(yearly_agg['year'], yearly_agg['avg_precipitation'])
slope_max, intercept_max, r_max, p_max, _ = stats.linregress(yearly_agg['year'], yearly_agg['max_precipitation'])

# ── Analysis 3: Extreme events ──
threshold_95    = df['pr'].quantile(0.95)
extreme_events  = df[df['pr'] > threshold_95]
extreme_by_year = extreme_events.groupby('year').size().reset_index(name='extreme_event_count')

print(f"All aggregations done in {time.perf_counter() - t2:.2f}s")


# ──────────────────────────────────────────────
# STEP 4 — PRINT SUMMARIES
# ──────────────────────────────────────────────

print("\n── Analysis 1: Spatial Aggregation ──")
print(spatial_agg.describe())

print("\n── Analysis 2: Temporal Trend ──")
print(yearly_agg.to_string(index=False))
print(f"\nAverage precip  slope={slope_avg:.4f}  R²={r_avg**2:.4f}  p={p_avg:.4f}")
print(f"Maximum precip  slope={slope_max:.4f}  R²={r_max**2:.4f}  p={p_max:.4f}")

print("\n── Analysis 3: Extreme Events ──")
print(f"95th-pct threshold: {threshold_95:.4f} mm")
print(f"Extreme events: {len(extreme_events):,} ({len(extreme_events)/len(df)*100:.2f}%)")
print(extreme_by_year.to_string(index=False))


# ──────────────────────────────────────────────
# STEP 5 — PLOTTING
# ──────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('CHIRPS Africa Precipitation Analysis 2012-2021', fontsize=14)
plt.tight_layout(pad=3.0)

# Plot 1: Spatial avg precipitation
ax = axes[0, 0]
sc = ax.scatter(spatial_agg['lon'], spatial_agg['lat'],
                c=spatial_agg['avg_precipitation'], cmap='YlOrRd', s=2, alpha=0.6)
plt.colorbar(sc, ax=ax, label='mm')
ax.set_title('Average Precipitation per (Lat, Lon)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Plot 2: Spatial max precipitation
ax = axes[0, 1]
sc = ax.scatter(spatial_agg['lon'], spatial_agg['lat'],
                c=spatial_agg['max_precipitation'], cmap='Blues', s=2, alpha=0.6)
plt.colorbar(sc, ax=ax, label='mm')
ax.set_title('Maximum Precipitation per (Lat, Lon)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Plot 3: Yearly average trend
ax = axes[1, 0]
ax.plot(yearly_agg['year'], yearly_agg['avg_precipitation'], marker='o', color='steelblue', label='Yearly Avg')
reg_line = slope_avg * yearly_agg['year'] + intercept_avg
ax.plot(yearly_agg['year'], reg_line, linestyle='--', color='red',
        label=f'Trend (slope={slope_avg:.4f}, R²={r_avg**2:.3f})')
ax.set_title('Yearly Average Precipitation')
ax.set_xlabel('Year')
ax.set_ylabel('Avg Precipitation (mm)')
ax.legend()

# Plot 4: Yearly max trend
ax = axes[1, 1]
ax.plot(yearly_agg['year'], yearly_agg['max_precipitation'], marker='s', color='darkorange', label='Yearly Max')
reg_line = slope_max * yearly_agg['year'] + intercept_max
ax.plot(yearly_agg['year'], reg_line, linestyle='--', color='red',
        label=f'Trend (slope={slope_max:.2f}, R²={r_max**2:.3f})')
ax.set_title('Yearly Maximum Precipitation')
ax.set_xlabel('Year')
ax.set_ylabel('Max Precipitation (mm)')
ax.legend()

# Plot 5: Extreme event counts per year
ax = axes[2, 0]
ax.bar(extreme_by_year['year'], extreme_by_year['extreme_event_count'], color='tomato', width=0.6)
ax.set_title(f'Extreme Events per Year (threshold = {threshold_95:.2f} mm)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Events')

# Plot 6: Precipitation distribution with threshold
ax = axes[2, 1]
ax.hist(df['pr'], bins=60, color='steelblue', edgecolor='white', log=True)
ax.axvline(threshold_95, color='red', linestyle='--', label=f'95th pct = {threshold_95:.2f} mm')
ax.set_title('Precipitation Distribution')
ax.set_xlabel('Precipitation (mm)')
ax.set_ylabel('Frequency (log scale)')
ax.legend()

plt.savefig('./precipitation_analysis_dask.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPlot saved to ./precipitation_analysis_dask.png")
print(f"Total wall-clock time: {time.perf_counter() - t0:.2f}s")