# HPC Project – WASCAL

This repository contains a High-Performance Computing (HPC) project developed in the context of climate data analysis and parallel computing experiments.

## 📌 Overview

The goal of this project is to explore and evaluate different parallelization strategies for processing large-scale datasets. It focuses on:

- Performance analysis (execution time, speedup, efficiency)
- Scalability with different numbers of CPU cores
- Comparison of parallel approaches (e.g., multiprocessing, Dask)

The project is inspired by the need for high computational power in climate science applications

## ⚙️ Features

- Parallel data processing using Python
- Performance evaluation across multiple core counts
- Visualization of performance metrics
- Analysis of scalability and optimal core usage
- spatial analysis
- temporal trend
- extreme event computation

## 🛠️ Technologies Used

- Python
- NumPy / Pandas / Xarray
- Matplotlib
- Multiprocessing (and optional Dask)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ahmadkongodev/hpc_project_wascal.git
cd hpc_project_wascal
```
### 2. Install dependencies
All dependencies are listed in the requirements.yaml file.
```
conda env create -f requirements.yaml
conda activate <env_name>
```
