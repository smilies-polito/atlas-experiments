import argparse
import os
import time

import muon as mu
import numpy as np
import pandas as pd
from atlas.tl import PalantirExtension

from supplementary.memory_benchmark.benchmark_pex import SEED, choose_early_cell


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("data_files", nargs="+")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--graph_key", default="wnn")
    p.add_argument("--n_components", type=int, default=10)
    p.add_argument("--knn", type=int, default=30)
    p.add_argument("--num_waypoints", type=int, default=1200)
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--root_celltype", default="HSC")
    p.add_argument("--cluster_key", default="cell_type")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def pseudotime_path(out_dir, data_file):
    tag = os.path.basename(data_file)[: -len(".h5mu")]
    return os.path.join(out_dir, f"pseudotime_{tag}.tsv")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for data_file in args.data_files:
        out = pseudotime_path(args.out_dir, data_file)
        if os.path.exists(out) and not args.overwrite:
            print(f"skip {out} (exists)")
            continue

        np.random.seed(SEED)
        start = time.perf_counter()
        data = mu.read_h5mu(data_file)
        data.pull_obs()
        early_cell = choose_early_cell(data, args.root_celltype, args.cluster_key)

        pex = PalantirExtension(mudata=data)
        pex.compute_kernel(key=args.graph_key)
        pex.compute_diffusion_maps(n_components=args.n_components, seed=SEED)
        pex.compute_multiscale_space()
        pex.run(early_cell=early_cell, knn=args.knn, num_waypoints=args.num_waypoints,
                n_jobs=args.n_jobs, random_state=SEED)

        pt = data.obs["pseudotime"]
        if pt.isna().any():
            raise ValueError(f"{data_file}: pseudotime has {pt.isna().sum()} missing values")
        df = pd.DataFrame({"pseudotime": pt.values, "early_cell": early_cell}, index=data.obs_names)
        df.index.name = "cell"
        tmp = f"{out}.tmp"
        df.to_csv(tmp, sep="\t")
        os.replace(tmp, out)
        print(f"{out}: {data.n_obs} cells, root {early_cell}, {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
