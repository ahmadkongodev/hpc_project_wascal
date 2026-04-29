import time
import pandas as pd
import xarray as xr
from multiprocessing import Pool
import matplotlib.pyplot as plt

# function to process each year's data and return the number of rows instead of loading the full dataframe to save memory and time during evaluation
def process_year(year):
    file_path = f"./All Data 2012 - 2021/CHIRPS_total_precipitation_day_0.25x0.25_africa_{year}_v2.0.nc"

    ds = xr.open_dataset(file_path)
    # Map to Normalize coordinate names 
    rename_map = {}
    if 'latitude' in ds.coords:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.coords:
        rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)

    df = ds.to_dataframe().dropna().reset_index()
    ds.close()
    return len(df)   #return number of rows instead of full dataframe to save memory and time during evaluation

# function to run the processing with different core counts and measure performance
def evaluator(core_counts):
    years = range(2012, 2022)
    results = [] # list to store results for each core count

    baseline_time = None

    for cores in core_counts:
        print(f"\nRunning with {cores} cores...")
        t0 = time.perf_counter()

        with Pool(processes=cores) as pool:
            output = pool.map(process_year, years)

        time_elapsed = time.perf_counter() - t0

        # Set baseline time for single core to calculate speedup and efficiency
        if baseline_time is None:
            baseline_time = time_elapsed

        # Calculate speedup and efficiency
        speedup = baseline_time / time_elapsed
        efficiency = speedup / cores

        results.append({
            "cores": cores,
            "time": time_elapsed,
            "speedup": speedup,
            "efficiency": efficiency
        })

        print(f"Time: {time_elapsed:.2f}s | Speedup: {speedup:.2f} | Efficiency: {efficiency:.2f}")
    
    # Convert results to DataFrame for better visualization
    return pd.DataFrame(results)


if __name__ == "__main__":
    core_counts = [1, 2, 3, 4, 5, 6, 7]  # stop at 7 because gettiing memory errors at 8 cores
    results_df = evaluator(core_counts)

    print("\nEvaluation Results:")
    print(results_df)

    # execution time plot
    plt.figure(figsize=(8,5))
    plt.plot(results_df["cores"], results_df["time"], marker='o')
    plt.xlabel("Number of Cores")
    plt.ylabel("Execution Time (s)")
    plt.title("Execution Time vs Core Count")
    plt.grid()
    plt.savefig('execution_time.png', dpi=400, bbox_inches='tight')
 
    # speedup plot
    plt.figure(figsize=(8,5))
    plt.plot(results_df["cores"], results_df["speedup"], marker='o')
    plt.plot(results_df["cores"], results_df["cores"], '--', label="Ideal Speedup")
    plt.xlabel("Number of Cores")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Core Count")
    plt.legend()
    plt.grid()
    plt.savefig('speedup.png', dpi=400, bbox_inches='tight')

    

    # efficiency plot
    plt.figure(figsize=(8,5))
    plt.plot(results_df["cores"], results_df["efficiency"], marker='o')
    plt.xlabel("Number of Cores")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Core Count")
    plt.grid()
    plt.savefig('efficiency.png', dpi=400, bbox_inches='tight')
 