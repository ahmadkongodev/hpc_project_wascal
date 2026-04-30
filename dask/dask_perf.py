import time
import pandas as pd
import xarray as xr
import dask
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt

# ==========================================
# Function to process one year (Dask version)
# ==========================================
def process_year_dask(year):
    file_path = f"./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc"

    # Open dataset lazily with chunking
    ds = xr.open_dataset(file_path, chunks={"time": 50})

    # Normalize coordinate names
    rename_map = {}
    if 'latitude' in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)

    # Avoid full dataframe conversion → use lazy count
    # Count valid (non-NaN) values directly
    var_name = list(ds.data_vars)[0]
    count = ds[var_name].count()

    return count


# ==========================================
# Evaluator using Dask cluster
# ==========================================
def evaluator_dask(core_counts):
    years = list(range(2012, 2022))
    results = []

    baseline_time = None

    for cores in core_counts:
        print(f"\nRunning with {cores} Dask workers...")

        # Create a local Dask cluster
        cluster = LocalCluster(
            n_workers=cores,
            threads_per_worker=1,
            memory_limit='2GB'   # adjust if needed
        )
        client = Client(cluster)

        t0 = time.perf_counter()

        # Create delayed tasks
        tasks = [dask.delayed(process_year_dask)(year) for year in years]

        # Trigger computation
        results_counts = dask.compute(*tasks)

        time_elapsed = time.perf_counter() - t0

        client.close()
        cluster.close()

        # Baseline (1 worker)
        if baseline_time is None:
            baseline_time = time_elapsed

        speedup = baseline_time / time_elapsed
        efficiency = speedup / cores

        results.append({
            "cores": cores,
            "time": time_elapsed,
            "speedup": speedup,
            "efficiency": efficiency
        })

        print(f"Time: {time_elapsed:.2f}s | Speedup: {speedup:.2f} | Efficiency: {efficiency:.2f}")

    return pd.DataFrame(results)


# ==========================================
# Main execution
# ==========================================
if __name__ == "__main__":
    core_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # test up to 10 workers

    results_df = evaluator_dask(core_counts)

    print("\nEvaluation Results:")
    print(results_df)

    # ==============================
    # Plotting
    # ==============================

    # Execution time
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["cores"], results_df["time"], marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Workers (Dask)")
    plt.grid()
    plt.savefig('plots/performance/execution_time_dask.png', dpi=400, bbox_inches='tight')

    # Speedup
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["cores"], results_df["speedup"], marker='o')
    plt.plot(results_df["cores"], results_df["cores"], '--', label="Ideal Speedup")
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Workers (Dask)")
    plt.legend()
    plt.grid()
    plt.savefig('plots/performance/speedup_dask.png', dpi=400, bbox_inches='tight')

    # Efficiency
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["cores"], results_df["efficiency"], marker='o')
    plt.xlabel("Number of Workers")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Workers (Dask)")
    plt.grid()
    plt.savefig('plots/performance/efficiency_dask.png', dpi=400, bbox_inches='tight')