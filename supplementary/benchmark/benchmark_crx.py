import argparse
import os
import sys
import traceback

import muon as mu
import numpy as np
import pandas as pd
from atlas.tl import CellRankExtension

from supplementary.memory_benchmark.benchmark_pex import SEED, find_obs_column, parse_pct
from supplementary.memory_benchmark.utils import benchmark_function, environment_info, write_row

STEPS = ("compute_kernel", "run", "pipeline")
PACKAGES = ("atlas", "cellrank", "pygpcca", "petsc4py", "slepc4py", "muon", "anndata", "scanpy",
            "numpy", "scipy", "scikit-learn", "pandas", "joblib")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--pseudotime_file", required=True)
    p.add_argument("--step", required=True, choices=STEPS)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--dataset", default="BMMC")
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--backend", default="loky")
    p.add_argument("--graph_key", default="wnn")
    p.add_argument("--cluster_key", default="cell_type")
    # kernel
    p.add_argument("--threshold_scheme", default="hard", choices=("hard", "soft"))
    p.add_argument("--frac_to_keep", type=float, default=0.3)
    # GPCCA
    p.add_argument("--n_components", type=int, default=20)
    p.add_argument("--n_states", type=int, default=10,
                   help="fixed number of macrostates, so that the work done is comparable "
                        "across dataset sizes (the eigengap heuristic would change it)")
    p.add_argument("--n_cells", type=int, default=30)
    p.add_argument("--allow_overlap", type=int, default=1, choices=(0, 1),
                   help="let initial and terminal states share cells; otherwise CellRank stops "
                        "with an error whenever they do, which can happen at some subset sizes only")
    p.add_argument("--solver", default="gmres")
    p.add_argument("--use_petsc", type=int, default=1, choices=(0, 1))
    return p.parse_args()


def require_petsc():
    try:
        import petsc4py  # noqa: F401
        import slepc4py  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"use_petsc=1 but PETSc/SLEPc is not importable: {exc}") from exc


def shutdown_workers():
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)


def load_pseudotime(data, path):
    pt = pd.read_csv(path, sep="\t", index_col="cell")
    missing = data.obs_names.difference(pt.index)
    if len(missing):
        raise ValueError(f"{path}: {len(missing)} cells of the dataset have no pseudotime")
    data.obs["pseudotime"] = pt.loc[data.obs_names, "pseudotime"].to_numpy()
    return str(pt["early_cell"].iloc[0])


def n_states(obs, column):
    if column not in obs:
        return np.nan
    return int(obs[column].dropna().nunique())


def build_steps(crx, args, cluster_column):
    def kernel():
        crx.compute_kernel(
            key=args.graph_key,
            time_key="pseudotime",
            cluster_key=cluster_column,
            threshold_scheme=args.threshold_scheme,
            frac_to_keep=args.frac_to_keep,
            n_jobs=args.n_jobs,
            backend=args.backend,
        )

    def run():
        crx.run(
            n_components=args.n_components,
            method="krylov",
            n_states=args.n_states,
            n_cells=args.n_cells,
            allow_overlap=bool(args.allow_overlap),
            solver=args.solver,
            use_petsc=bool(args.use_petsc),
            n_jobs=args.n_jobs,
            backend=args.backend,
        )

    def pipeline():
        kernel()
        run()

    return {
        "compute_kernel": ([], kernel),
        "run": ([kernel], run),
        "pipeline": ([], pipeline),
    }


def main():
    args = parse_args()
    np.random.seed(SEED)

    data = mu.read_h5mu(args.data_file)
    data.pull_obs()
    early_cell = load_pseudotime(data, args.pseudotime_file)
    cluster_column = find_obs_column(data, args.cluster_key)

    crx = CellRankExtension(mudata=data)
    prerequisites, target = build_steps(crx, args, cluster_column)[args.step]

    row = {
        "dataset": args.dataset,
        "data_file": os.path.basename(args.data_file),
        "pct": parse_pct(args.data_file),
        "n_cells": data.n_obs,
        "rep": args.rep,
        "method": "CellRankExtension",
        "function": f"crx.{args.step}",
        "n_jobs": args.n_jobs,
        "backend": args.backend,
        "seed": SEED,
        "graph_key": args.graph_key,
        "pseudotime_file": os.path.basename(args.pseudotime_file),
        "pseudotime_root": early_cell,
        "threshold_scheme": args.threshold_scheme,
        "frac_to_keep": args.frac_to_keep,
        "n_components": args.n_components,
        "n_states_requested": args.n_states,
        "n_cells_per_state": args.n_cells,
        "allow_overlap": bool(args.allow_overlap),
        "solver": args.solver,
        "use_petsc": bool(args.use_petsc),
        **environment_info(packages=PACKAGES),
    }

    status = 0
    try:
        if args.use_petsc:
            require_petsc()
        for step in prerequisites:
            step()
        if prerequisites:
            shutdown_workers()
        _, metrics = benchmark_function(target)
        row.update(metrics, status="ok", error="")
        if args.step in ("run", "pipeline"):
            row.update(
                n_macrostates=n_states(data.obs, "macrostates"),
                n_initial_states=n_states(data.obs, "initial_states"),
                n_terminal_states=n_states(data.obs, "terminal_states"),
            )
    except Exception as exc:  
        traceback.print_exc()
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        status = 1

    write_row(row, args.out_file)
    return status


if __name__ == "__main__":
    sys.exit(main())
