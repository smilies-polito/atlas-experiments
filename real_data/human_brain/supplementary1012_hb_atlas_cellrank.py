########################################################################
# This code applies ATLAS TI on the Human Fetal Brain data             #
# Plots can be visualized in Supp Fig 10 and Supp Fig 12               #
########################################################################

import os
import atlas 
import argparse
import numpy as np
import pandas as pd
import muon as mu
from real_data.utils import _compute_results, _get_plots, _save_metrics
from atlas.tl import CellRankExtension


if __name__=="__main__":
    seed = 42 
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_rna", type=int, default=15)
    parser.add_argument("--knn_act", type=int, default=15)
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--wnn", default=None)
    parser.add_argument("--n_states", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--removed", type=int, default=0)
    args=parser.parse_args()

    removed = bool(args.removed)
    prefix = "removed" if removed else "non_removed" 

    pCode = f"{prefix}_{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}"
    code = f"cellrank_{pCode}_{args.n_states}"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "human_brain")
    result_path = os.path.join(output_dir, "results.tsv")
    data_path = os.path.join(output_dir, f"{pCode}.h5mu")

    mudata = mu.read_h5mu(data_path)
    pseudotime = pd.read_csv(os.path.join(output_dir, 
                        f"pseudotime_{pCode}.tsv"),
                        sep = "\t", index_col = 0, header = 0)
    mudata.obs = mudata.obs.merge(pseudotime, 
                                    how = "left",
                                    left_index=True, 
                                    right_index=True)

    cex = CellRankExtension(mudata=mudata)
    cex.compute_kernel(key = "wnn",
                    time_key = "pseudotime",
                    cluster_key = "Cluster.Name",
                    n_jobs = args.n_jobs)

    try: 
        cex.run(n_states = args.n_states, 
             use_petsc = True,
             allow_overlap = False,
             n_jobs = args.n_jobs)
        allow_overlap = False

    except Exception as e:
        print(f"allow_overlap = False gave Exception:\n{e}\n. Trying with allow_overlap=True.")
        cex.run(n_states = args.n_states, 
            use_petsc = True,
            allow_overlap = True,
            n_jobs = args.n_jobs)
        allow_overlap = True

    # Save if allow overlap has been used or not 
    mudata.uns["atlas"] = {"allow_overlap": allow_overlap}

    print("AVERAGE PSEUDOTIME PER MACROSTATE:")
    print(mudata.obs[["macrostates", "pseudotime"]].groupby("macrostates").mean())

    metrics = _compute_results(mudata = mudata,
                                    code = code,
                                    seed = seed)
    _save_metrics(metrics, result_path)
    print("METRICS")
    print(metrics)

    _get_plots(mudata = mudata,
            cluster_key = "Cluster.Name",
            code = code,
            seed = seed)

    mudata.write(os.path.join(output_dir, 
                                f"{code}.h5mu")
                )
