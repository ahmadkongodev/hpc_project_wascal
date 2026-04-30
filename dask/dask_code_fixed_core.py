import os
import time
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from dask.diagnostics import ProgressBar

from dask.distributed import Client, LocalCluster

# ==============================
# OUTPUT FOLDER
# ==============================
output_dir = "./dask_plots"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    t0 = time.perf_counter()

    # ──────────────────────────────────────────────
    # HPC CLUSTER (FIXED: 5 CORES)
    # ──────────────────────────────────────────────
    n_cores = 5

    cluster = LocalCluster(
        n_workers=n_cores,
        threads_per_worker=1,
        processes=True,
        memory_limit="2GB"
    )

    client = Client(cluster)
    print(client)

    # ──────────────────────────────────────────────
    # DATA LOADING (DASK)
    # ──────────────────────────────────────────────
    print("Loading data with Dask...")

    files = "./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_*.nc"

    def preprocess(ds):
        if 'latitude' in ds.coords:
            ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
        return ds

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        preprocess=preprocess,
        chunks={"time": 365}
    )

    pr = ds['pr']
    print("Dataset loaded (lazy).")

    # ──────────────────────────────────────────────
    # 1. SPATIAL AGGREGATION
    # ──────────────────────────────────────────────
    print("\n── Analysis 1: Spatial Aggregation ──")

    spatial_agg = xr.Dataset({
        "avg_precipitation": pr.mean(dim="time"),
        "max_precipitation": pr.max(dim="time")
    })

    with ProgressBar():
        spatial_agg = spatial_agg.compute()

    spatial_df = spatial_agg.to_dataframe().reset_index()

    # ──────────────────────────────────────────────
    # 2. TEMPORAL TREND
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
    # 3. EXTREME EVENTS
    # ──────────────────────────────────────────────
    print("\n── Analysis 3: Extreme Events ──")

    threshold_95 = pr.quantile(0.95, dim=["time", "lat", "lon"])

    with ProgressBar():
        threshold_95 = threshold_95.compute()

    extreme_mask = pr > threshold_95

    with ProgressBar():
        extreme_count = extreme_mask.sum().compute()

    print(f"95th percentile threshold: {float(threshold_95.values):.4f}")
    print(f"Total extreme events: {int(extreme_count.values):,}")

    # ──────────────────────────────────────────────
    # PLOTTING
    # ──────────────────────────────────────────────
    print("\n── Generating plots ──")

    # Plot 1
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        spatial_df['lon'], spatial_df['lat'],
        c=spatial_df['avg_precipitation'],
        cmap='Greens', s=2, alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label='mm')
    ax.set_title('Average Precipitation per (Lat, Lon)')
    plt.savefig(os.path.join(output_dir, 'plot1.png'))
    plt.close()

    # Plot 2
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        spatial_df['lon'], spatial_df['lat'],
        c=spatial_df['max_precipitation'],
        cmap='Blues', s=2, alpha=0.6
    )
    plt.colorbar(sc, ax=ax, label='mm')
    ax.set_title('Maximum Precipitation per (Lat, Lon)')
    plt.savefig(os.path.join(output_dir, 'plot2.png'))
    plt.close()

    # Plot 3
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(yearly_df['year'], yearly_df['avg_precipitation'], marker='o')

    reg_line = slope_avg * yearly_df['year'] + intercept_avg
    ax.plot(yearly_df['year'], reg_line, '--', color='red')

    plt.savefig(os.path.join(output_dir, 'plot3.png'))
    plt.close()

    # Plot 4 (extremes per year)
    extreme_by_year = extreme_mask.groupby("time.year").sum(dim=["time", "lat", "lon"])

    with ProgressBar():
        extreme_by_year = extreme_by_year.compute()

    extreme_df = extreme_by_year.to_dataframe(name="extreme_event_count").reset_index()

    fig, ax = plt.subplots()
    ax.bar(extreme_df['year'], extreme_df['extreme_event_count'])
    plt.savefig(os.path.join(output_dir, 'plot4.png'))
    plt.close()

    # ──────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────
    print(f"\nTotal time: {time.perf_counter() - t0:.2f}s")
    print("Done with 5-core Dask experiment.")