# Computational benchmark of ATLAS

Runtime and memory of the two ATLAS trajectory-inference strategies,
`atlas.tl.PalantirExtension` and `atlas.tl.CellRankExtension`, measured on
increasingly large subsets of the human bone marrow multiome dataset
(NCBI GEO GSE194122).

Every measurement runs in its own process, inside the same container, pinned to
the same CPU cores. The steps that a measurement depends on run first and are not
timed. Each configuration is repeated several times, in random order.

All commands below are run from the repository root (the folder that contains
`supplementary/`).

## Contents

| File | Purpose |
|---|---|
| `utils.py` | Measures one call: wall time, CPU time, peak memory of the main process (exact, from the kernel high-water mark) and of worker processes (PSS), software and hardware context |
| `memory_monitor.py` | Separate process that samples the memory of worker processes during a call; started by `utils.py` |
| `benchmark_pex.py` | Measures one step of `PalantirExtension` on one dataset, one replicate |
| `run_benchmark_pex.sh` | Runs the whole Palantir benchmark |
| `prepare_crx_pseudotime.py` | Computes, once per dataset, the pseudotime used as input by `CellRankExtension` |
| `benchmark_crx.py` | Measures one step of `CellRankExtension` on one dataset, one replicate |
| `run_benchmark_crx.sh` | Runs the whole CellRank benchmark |
| `analyze_benchmark.py` | Summary statistics, scaling exponents, check of the server load |
| `plot_benchmark.py` | LaTeX tables (and, optionally, a figure) from the results |
| `diagnose_diffusion_maps.py` | Counts the ARPACK iterations of the diffusion maps at each size |
| `diagnose_dm_threads.py` | Times the diffusion maps with and without BLAS multithreading |

## Requirements

- Linux (the measurements use `/proc`, `taskset` and the kernel memory high-water mark)
- Apptainer or Singularity
- The container image with ATLAS and its dependencies, including `psutil`,
  `threadpoolctl`, `petsc4py` and `slepc4py` (`container_new.sif` below)

The launchers call Python only inside the container, so no Python installation is
needed on the host.

## 1. Input data

The benchmark reads `output/benchmark/data_<fraction>.h5mu`, one file per subset,
for example `data_0.1.h5mu`, …, `data_1.0.h5mu`. Each file is a MuData object with
gene expression and ATAC-derived gene activity, a cell-type annotation
(`cell_type`) and the WNN graph computed by ATLAS (`atlas.pp.wnn`, key `wnn`), 
see `subset_dataset.py` to understand how the dataset was obtained.

## 2. Palantir-based strategy

```bash
SIF=container_new.sif CORES=0-7 N_REPS=7 \
  bash supplementary/benchmark/run_benchmark_pex.sh 2>&1 | tee bench_pex.log
```

This runs five steps (`compute_kernel`, `compute_diffusion_maps`,
`compute_multiscale_space`, `run` and `pipeline`, which runs all four in sequence)
for every dataset and replicate. With five datasets and seven replicates that is 175
runs. Results are written to `output/benchmark/results_pex/`.

## 3. CellRank-based strategy

**3a. Pseudotime.** The CellRank pseudotime kernel needs a pseudotime. It is computed
once per dataset with `PalantirExtension`, using the same root cell, seed and
parameters as the Palantir benchmark, and saved to `output/benchmark/pseudotime/`:

```bash
APPTAINERENV_PYTHONPATH=$PWD APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 \
  taskset -c 0-7 singularity exec --cleanenv PATH_TO_CONTAINER \
  python3 -m supplementary.benchmark.prepare_crx_pseudotime \
  --out_dir output/benchmark/pseudotime output/benchmark/data_*.h5mu
```

**3b. Full benchmark.** Three steps (`compute_kernel`, `run` and `pipeline`) for every
dataset and replicate, 105 runs. Results are written to
`output/benchmark/results_crx/`.

```bash
SIF=container_new.sif CORES=0-7 N_REPS=7 \
  bash supplementary/benchmark/run_benchmark_crx.sh 2>&1 | tee bench_crx.log
```

If `petsc4py` or `slepc4py` is missing, CellRank silently falls back to SciPy solvers
and to a dense Schur decomposition, which changes the algorithm and needs memory
quadratic in the number of cells. The launcher and `benchmark_crx.py` therefore stop
instead of running without PETSc.

## 4. Analysis and tables

```bash
for m in pex crx; do
  singularity exec --cleanenv PATH_TO_CONTAINER python3 -m supplementary.benchmark.analyze_benchmark \
    output/benchmark/results_${m}/resources_${m}.tsv
done
```

## Outputs

In `output/benchmark/results_pex/` and `output/benchmark/results_crx/`:

| File | Content |
|---|---|
| `resources_*.tsv` | All measurements, one row per run: dataset, step, replicate, parameters, wall and CPU time, memory, software versions, container checksum, CPU cores, server load |
| `summary_resources_*.tsv` | Per step and dataset size: median, quartiles and coefficient of variation of time and memory |
| `scaling_resources_*.tsv` | Log-log scaling exponents with bootstrap 95% confidence intervals, on all runs and on runs started under below-median server load |
| `failures.tsv` | Runs that ended with an error, a timeout (exit code 124) or were killed (137, usually out of memory) |
| `raw/`, `logs/` | One result file and one log per run |

Memory is reported above the baseline of the loaded data. For the CellRank strategy it
is split into the main process, which runs the algorithm, and the worker processes
started for `n_jobs`. The launcher resumes where it stopped: runs with a result file
are skipped, and runs that failed are repeated.

## Settings

Both launchers are configured through environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `SIF` | required | Container image |
| `CORES` | `0-15` | CPUs the runs are pinned to (`0-7` in the paper) |
| `N_REPS` | `5` | Replicates per step and dataset (`7` in the paper) |
| `N_JOBS` | number of CPUs in `CORES` | `n_jobs` passed to ATLAS |
| `BLAS_THREADS` | `1` | Threads of BLAS/OpenMP libraries; `n_jobs` is the only parallelism |
| `STEPS` | all steps | Steps to measure |
| `DATA_DIR` | `output/benchmark` | Folder with the `data_*.h5mu` files |
| `OUT_DIR` | `output/benchmark/results_pex` or `results_crx` | Results folder |
| `RUN_TIMEOUT` | `6h` | Time limit of a single run |
| `MAX_LOAD` | empty | If set, wait before each run until the 1-minute load average is below it |
| `ORDER_SEED` | `2026` | Seed of the random run order |
| `CLUSTER_KEY` | `cell_type` | Cell-type column |
| `ROOT_CELLTYPE` | `HSC` | Cell type of the root cell (Palantir only) |
| `PSEUDOTIME_DIR` | `$DATA_DIR/pseudotime` | Pseudotime files (CellRank only) |
| `N_STATES` | `10` | Number of GPCCA macrostates (CellRank only) |
| `N_COMPONENTS` | `20` | Number of Schur vectors (CellRank only) |
| `USE_PETSC` | `1` | Use PETSc/SLEPc (CellRank only) |
| `ALLOW_OVERLAP` | `1` | Let initial and terminal states share cells (CellRank only) |
| `PYTHON` | detected | Python interpreter inside the container |
| `EXTRA_BINDS` | empty | Extra folders to mount in the container, comma-separated |

## Known issue

For the 51,936-cell subset, CellRank could not compute the coarse-grained stationary
distribution needed for automatic initial-state prediction: the solver did not
converge on the nearly reducible transition matrix produced by the hard-thresholded
pseudotime kernel. 

## Hardware used in the simulations

AMD Ryzen 9 5950X (16 cores, 32 threads, two core complexes with 32 MB L3 cache each),
64 GB RAM, Ubuntu 22.04.5 LTS, Apptainer 1.4.5. The server was shared with other
users. The exact software versions and the checksum of the container are recorded in
every row of `resources_*.tsv`.
