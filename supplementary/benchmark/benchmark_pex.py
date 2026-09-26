import argparse
import os
import re
import sys
import traceback
import warnings

import muon as mu
import numpy as np
from atlas.tl import PalantirExtension

from supplementary.memory_benchmark.utils import benchmark_function, environment_info, write_row

SEED = 42
STEPS = ("compute_kernel", "compute_diffusion_maps", "compute_multiscale_space", "run", "pipeline")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--step", required=True, choices=STEPS)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--dataset", default="BMMC")
    p.add_argument("--n_jobs", type=int, default=16)
    p.add_argument("--graph_key", default="wnn")
    p.add_argument("--n_components", type=int, default=10)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--num_waypoints", type=int, default=1200)
    p.add_argument("--root_celltype", default="HSC")
    p.add_argument("--cluster_key", default="cell_type")
    return p.parse_args()


def parse_pct(path):
    m = re.search(r"data_([0-9]+(?:\.[0-9]+)?)\.h5mu$", os.path.basename(path))
    return float(m.group(1)) if m else float("nan")


def find_obs_column(data, key):
    if key in data.obs.columns:
        return key
    matches = [c for c in data.obs.columns if c.split(":")[-1] == key]
    return matches[0] if matches else None


def choose_early_cell(data, celltype, cluster_key):
    rng = np.random.default_rng(SEED)
    column = find_obs_column(data, cluster_key)
    if column is not None and (data.obs[column].astype(str) == celltype).any():
        candidates = np.sort(data.obs_names[(data.obs[column].astype(str) == celltype).values])
    else:
        warnings.warn(f"'{celltype}' not found in obs['{cluster_key}']: drawing the root among all cells")
        candidates = np.sort(data.obs_names)
    return str(rng.choice(candidates))


def build_steps(pex, args, early_cell):
    def kernel():
        pex.compute_kernel(key=args.graph_key)

    def diffusion_maps():
        pex.compute_diffusion_maps(n_components=args.n_components, seed=SEED)

    def multiscale():
        pex.compute_multiscale_space()

    def run():
        pex.run(
            early_cell=early_cell,
            knn=args.knn,
            num_waypoints=args.num_waypoints,
            n_jobs=args.n_jobs,
            random_state=SEED,
        )

    def pipeline():
        kernel()
        diffusion_maps()
        multiscale()
        run()

    return {
        "compute_kernel": ([], kernel),
        "compute_diffusion_maps": ([kernel], diffusion_maps),
        "compute_multiscale_space": ([kernel, diffusion_maps], multiscale),
        "run": ([kernel, diffusion_maps, multiscale], run),
        "pipeline": ([], pipeline),
    }


def main():
    args = parse_args()
    np.random.seed(SEED)

    data = mu.read_h5mu(args.data_file)
    data.pull_obs()
    early_cell = choose_early_cell(data, args.root_celltype, args.cluster_key)

    pex = PalantirExtension(mudata=data)
    prerequisites, target = build_steps(pex, args, early_cell)[args.step]

    row = {
        "dataset": args.dataset,
        "data_file": os.path.basename(args.data_file),
        "pct": parse_pct(args.data_file),
        "n_cells": data.n_obs,
        "rep": args.rep,
        "method": "PalantirExtension",
        "function": f"pex.{args.step}",
        "n_jobs": args.n_jobs,
        "seed": SEED,
        "graph_key": args.graph_key,
        "n_components": args.n_components,
        "knn": args.knn,
        "num_waypoints": args.num_waypoints,
        "early_cell": early_cell,
        **environment_info(),
    }

    status = 0
    try:
        for step in prerequisites:
            step()
        _, metrics = benchmark_function(target)
        row.update(metrics, status="ok", error="")
    except Exception as exc:  # recorded, so failures are visible in the results table
        traceback.print_exc()
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
        status = 1

    write_row(row, args.out_file)
    return status


if __name__ == "__main__":
    sys.exit(main())
