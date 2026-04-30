import os
import time
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from multiprocessing import Pool

# ==============================
# Create output folder
# ==============================
output_dir = "./plots"

    
# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

#function to process each year's data and return it as a dataframe, this is used in the multiprocessing pool to load data in parallel
def process_year(year):
    print(f"Started processing file: CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc")
    file_path = f"./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc"

    ds = xr.open_dataset(file_path)
    # Some datasets may use 'latitude' and 'longitude' instead of 'lat' and 'lon'. We check for these coordinate names and rename them to a consistent format if necessary. This ensures that the rest of the code can work with a standardized set of coordinate names.
    rename_map = {}
    if 'latitude'  in ds.coords: rename_map['latitude']  = 'lat'
    if 'longitude' in ds.coords: rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)

    df = ds.to_dataframe().dropna().reset_index()
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year

    ds.close()
    print(f"Finished processing file: CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc")
    return df


# Main execution block
if __name__ == "__main__":
    years = range(2012, 2022)
    t0 = time.perf_counter()

    with Pool(processes=2) as pool: # 2 because this was giving the best performance , more cores were less efficient
        data_list = pool.map(process_year, years)
    # After the multiprocessing pool completes, we concatenate the list of dataframes into a single dataframe called full_data. 
    full_data = pd.concat(data_list, ignore_index=True)
    print(f"Data loading time: {time.perf_counter() - t0:.2f} seconds")
    print(f"Total rows: {len(full_data)}")

    pr_col = 'pr'  # column name for precipitation in the dataset

    # ──────────────────────────────────────────────
    # 1. DATA AGGREGATION
    # ──────────────────────────────────────────────

    print("\n── Analysis 1: Spatial Aggregation ──")

    spatial_agg = (
        full_data
        .groupby(['lat', 'lon'])[pr_col]
        .agg(avg_precipitation='mean', max_precipitation='max')
        .reset_index()
    )
    print(spatial_agg.describe())


    # ──────────────────────────────────────────────
    # 2. TEMPORAL TREND
    # ──────────────────────────────────────────────

    print("\n── Analysis 2: Temporal Trend ──")

    yearly_agg = (
        full_data
        .groupby('year')[pr_col]
        .agg(avg_precipitation='mean', max_precipitation='max')
        .reset_index()
    )

    # Perform linear regression to identify trends over time
    # We calculate the slope, intercept, correlation coefficient (r), and p-value for both average and maximum precipitation trends.
    slope_avg, intercept_avg, r_avg, p_avg, _ = stats.linregress(yearly_agg['year'], yearly_agg['avg_precipitation'])
    # R² is the square of the correlation coefficient (r)
    slope_max, intercept_max, r_max, p_max, _ = stats.linregress(yearly_agg['year'], yearly_agg['max_precipitation'])

    print(yearly_agg.to_string(index=False)) #index=False to print without the index column
    print(f"\nAverage precip  slope={slope_avg:.4f}  R²={r_avg**2:.4f}  p={p_avg:.4f}")
    print(f"Maximum precip  slope={slope_max:.4f}  R²={r_max**2:.4f}  p={p_max:.4f}")


    # ──────────────────────────────────────────────
    # 3. EXTREME EVENTS
    # ──────────────────────────────────────────────

    print("\n── Analysis 3: Extreme Events ──")
    # We define extreme events as those where the precipitation exceeds the 95th percentile threshold. We calculate this threshold from the entire dataset and then count how many events exceed it, both in total and by year.
    threshold_95    = full_data[pr_col].quantile(0.95)
    # extreme_events in total
    extreme_events  = full_data[full_data[pr_col] > threshold_95]
    # extreme events by year
    extreme_by_year = extreme_events.groupby('year').size().reset_index(name='extreme_event_count')

    print(f"95th-pct threshold: {threshold_95:.4f} mm")
    print(f"Extreme events: {len(extreme_events):,} ({len(extreme_events)/len(full_data)*100:.2f}%)")
    print(extreme_by_year.to_string(index=False))


    # ──────────────────────────────────────────────
    # PLOTTING
    # ──────────────────────────────────────────────

        
    # ==============================
    # Plot 1: Spatial avg precipitation
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        spatial_agg['lon'], spatial_agg['lat'],
        c=spatial_agg['avg_precipitation'],
        cmap='Greens', s=2, alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label='mm')
    ax.set_title('Average Precipitation per (Lat, Lon)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'plot1_avg_precipitation.png'),
                dpi=400, bbox_inches='tight')
    plt.close()

    # ==============================
    # Plot 2: Spatial max precipitation
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        spatial_agg['lon'], spatial_agg['lat'],
        c=spatial_agg['max_precipitation'],
        cmap='Blues', s=2, alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label='mm')
    ax.set_title('Maximum Precipitation per (Lat, Lon)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, 'plot2_max_precipitation.png'),
                dpi=400, bbox_inches='tight')
    plt.close()

    # ==============================
    # Plot 3: Yearly average trend
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        yearly_agg['year'],
        yearly_agg['avg_precipitation'],
        marker='o',
        color='steelblue',
        label='Yearly Avg'
    )
    reg_line = slope_avg * yearly_agg['year'] + intercept_avg
    ax.plot(
        yearly_agg['year'],
        reg_line,
        linestyle='--',
        color='red',
        label=f'Trend (slope={slope_avg:.4f}, R²={r_avg**2:.3f})'
    )
    ax.set_title('Yearly Average Precipitation')
    ax.set_xlabel('Year')
    ax.set_ylabel('Avg Precipitation (mm)')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'plot3_yearly_avg_trend.png'),
                dpi=400, bbox_inches='tight')
    plt.close()

    # ==============================
    # Plot 4: Yearly max trend
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        yearly_agg['year'],
        yearly_agg['max_precipitation'],
        marker='s',
        color='darkorange',
        label='Yearly Max'
    )
    reg_line = slope_max * yearly_agg['year'] + intercept_max
    ax.plot(
        yearly_agg['year'],
        reg_line,
        linestyle='--',
        color='red',
        label=f'Trend (slope={slope_max:.2f}, R²={r_max**2:.3f})'
    )
    ax.set_title('Yearly Maximum Precipitation')
    ax.set_xlabel('Year')
    ax.set_ylabel('Max Precipitation (mm)')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'plot4_yearly_max_trend.png'),
                dpi=400, bbox_inches='tight')
    plt.close()

    # ==============================
    # Plot 5: Extreme event counts per year
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        extreme_by_year['year'],
        extreme_by_year['extreme_event_count'],
        color='tomato',
        width=0.6
    )
    ax.set_title(f'Extreme Events per Year (threshold = {threshold_95:.2f} mm)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Events')
    plt.savefig(os.path.join(output_dir, 'plot5_extreme_events.png'),
                dpi=400, bbox_inches='tight')
    plt.close()

    # ==============================
    # Plot 6: Precipitation distribution
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        full_data[pr_col],
        bins=60,
        color='steelblue',
        edgecolor='white',
        log=True
    )
    ax.axvline(
        threshold_95,
        color='red',
        linestyle='--',
        label=f'95th pct = {threshold_95:.2f} mm'
    )
    ax.set_title('Precipitation Distribution')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Frequency (log scale)')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'plot6_precip_distribution.png'),
                dpi=400, bbox_inches='tight')
    plt.close()

    # ==============================
    # Done
    # ==============================
    print(f"All plots saved successfully in: {output_dir}")
    print(f"Total wall-clock time: {time.perf_counter() - t0:.2f}s")