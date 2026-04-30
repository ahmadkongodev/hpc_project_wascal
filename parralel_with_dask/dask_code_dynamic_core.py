import os
import time
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from dask.diagnostics import ProgressBar

# ==============================
# Create output folder
# ==============================
output_dir = "./dask_plots"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    t0 = time.perf_counter()

    # ──────────────────────────────────────────────
    # 1. LOAD DATA (DASK PARALLEL)
    # ──────────────────────────────────────────────

    print("Loading data with Dask...")

    files = "./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_*.nc"

    def preprocess(ds):
        # Only fix the problematic case
        if 'latitude' in ds.coords:
            ds = ds.rename({
                'latitude': 'lat',
                'longitude': 'lon'
            })
        return ds


    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        chunks={"time": 365},
        preprocess=preprocess
    )


    # Add year coordinate
    ds['year'] = ds['time'].dt.year

    pr = ds['pr']  # precipitation variable

    print("Dataset loaded (lazy).")

    # ──────────────────────────────────────────────
    # 2. SPATIAL AGGREGATION
    # ──────────────────────────────────────────────

    print("\n── Analysis 1: Spatial Aggregation ──")

    spatial_agg = xr.Dataset({
        "avg_precipitation": pr.mean(dim="time"),
        "max_precipitation": pr.max(dim="time")
    })

    with ProgressBar():
        spatial_agg = spatial_agg.compute()

    spatial_df = spatial_agg.to_dataframe().reset_index()
    print(spatial_df.describe())

    # ──────────────────────────────────────────────
    # 3. TEMPORAL TREND
    # ──────────────────────────────────────────────

    print("\n── Analysis 2: Temporal Trend ──")

    yearly_mean = pr.groupby("time.year").mean(dim=["time", "lat", "lon"])
    yearly_max  = pr.groupby("time.year").max(dim=["time", "lat", "lon"])

    yearly_agg = xr.Dataset({
        "avg_precipitation": yearly_mean,
        "max_precipitation": yearly_max
    })

    with ProgressBar():
        yearly_agg = yearly_agg.compute()

    yearly_df = yearly_agg.to_dataframe().reset_index()

    slope_avg, intercept_avg, r_avg, p_avg, _ = stats.linregress(
        yearly_df['year'], yearly_df['avg_precipitation']
    )

    slope_max, intercept_max, r_max, p_max, _ = stats.linregress(
        yearly_df['year'], yearly_df['max_precipitation']
    )

    print(yearly_df.to_string(index=False))

    # ──────────────────────────────────────────────
    # 4. EXTREME EVENTS
    # ──────────────────────────────────────────────
    print("\n── Analysis 3: Extreme Events ──")


    # This avoids collapsing all dimensions and prevents memory explosion
    threshold_95 = pr.quantile(0.95, dim="time")

    # Trigger computation safely
    with ProgressBar():
        threshold_95 = threshold_95.compute()

    extreme_mask = pr > threshold_95

    # Count total number of extreme events
    with ProgressBar():
        extreme_count = extreme_mask.sum().compute()

    # ----------------------------------
    # 3. Display results
    # ----------------------------------
    print(f"Example threshold value: {float(threshold_95.mean().values):.4f}")
    print(f"Total extreme events: {int(extreme_count.values):,}")
    # ──────────────────────────────────────────────
    # PLOTTING (same logic)
    # ──────────────────────────────────────────────
    
    print("\n── Generating plots ──")

    # ==============================
    # Plot 1: Spatial average precipitation
    # ==============================
    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(
        spatial_df['lon'], spatial_df['lat'],
        c=spatial_df['avg_precipitation'],
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
        spatial_df['lon'], spatial_df['lat'],
        c=spatial_df['max_precipitation'],
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
        yearly_df['year'],
        yearly_df['avg_precipitation'],
        marker='o',
        color='steelblue',
        label='Yearly Avg'
    )

    reg_line_avg = slope_avg * yearly_df['year'] + intercept_avg

    ax.plot(
        yearly_df['year'],
        reg_line_avg,
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
        yearly_df['year'],
        yearly_df['max_precipitation'],
        marker='s',
        color='darkorange',
        label='Yearly Max'
    )

    reg_line_max = slope_max * yearly_df['year'] + intercept_max

    ax.plot(
        yearly_df['year'],
        reg_line_max,
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
    # Plot 5: Extreme events per year
    # ==============================
    print("Computing extreme events per year...")

    extreme_by_year = extreme_mask.groupby("time.year").sum(dim=["time", "lat", "lon"])

    with ProgressBar():
        extreme_by_year = extreme_by_year.compute()

    extreme_df = extreme_by_year.to_dataframe(name="extreme_event_count").reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        extreme_df['year'],
        extreme_df['extreme_event_count'],
        color='tomato',
        width=0.6
    )

    ax.set_title('Extreme Events per Year (95th percentile)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Events')

    plt.savefig(os.path.join(output_dir, 'plot5_extreme_events.png'),
                dpi=400, bbox_inches='tight')
    plt.close()


    # ==============================
    # Plot 6: Precipitation distribution (SAFE SAMPLING)
    # ==============================
    print("Generating precipitation distribution plot...")

    # Sample only 1 year to avoid memory explosion
    sample = pr.isel(time=slice(0, 365)).values.flatten()
    sample = sample[~np.isnan(sample)]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        sample,
        bins=60,
        color='steelblue',
        edgecolor='white',
        log=True
    )

    threshold_value = float(threshold_95.mean().values)

    ax.axvline(
        threshold_value,
        color='red',
        linestyle='--',
        label=f'95th percentile ≈ {threshold_value:.2f} mm'
    )

    ax.set_title('Precipitation Distribution (Sampled)')
    ax.set_xlabel('Precipitation (mm)')
    ax.set_ylabel('Frequency (log scale)')
    ax.legend()

    plt.savefig(os.path.join(output_dir, 'plot6_precip_distribution.png'),
                dpi=400, bbox_inches='tight')
    plt.close()


    # ==============================
    # DONE
    # ==============================
    print(f"\nAll plots saved in: {output_dir}")
    print(f"Total wall-clock time: {time.perf_counter() - t0:.2f}s")
