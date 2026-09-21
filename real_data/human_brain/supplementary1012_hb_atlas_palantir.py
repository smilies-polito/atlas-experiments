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
from atlas.tl import PalantirExtension


if __name__=="__main__":
    seed = 42 
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default = 1)
    parser.add_argument("--knn_rna", type=int, default = 15)
    parser.add_argument("--knn_act", type=int, default = 15)
    parser.add_argument("--n_pcs_rna", type=int, default = 20)
    parser.add_argument("--n_pcs_act", type=int, default = 10)
    parser.add_argument("--wnn", default = None)
    parser.add_argument("--removed", type=int, default=0)
    args = parser.parse_args()

    removed = bool(args.removed)
    prefix = "removed" if removed else "non_removed" 
    pCode = f"{prefix}_{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}"
    code = f"palantir_{pCode}"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "human_brain")
    result_path = os.path.join(output_dir, "results.tsv")
    data_path = os.path.join(output_dir, f"{pCode}.h5mu")

    mudata = mu.read_h5mu(data_path)
    initial = rng.choice(mudata[mudata.obs["Cluster.Name"]=="Cyc. Prog."].obs_names)

    pex = PalantirExtension(mudata = mudata)
    pex.compute_kernel(key="wnn")
    pex.compute_diffusion_maps(seed = seed)
    pex.compute_multiscale_space()

    pex.run(early_cell = initial,
                n_jobs = args.n_jobs,
                random_state = seed)

    print("AVERAGE PSEUDOTIME PER CELL TYPE:")
    print(mudata.obs[["pseudotime", "Cluster.Name"]].groupby("Cluster.Name").mean())

    print("AVERAGE PSEUDOTIME PER TERMINAL STATE:")
    print(mudata.obs[["terminal_states", "pseudotime"]].groupby("terminal_states").mean())

    print("AVERAGE PSEUDOTIME PER INITIAL STATE:")
    print(mudata.obs[["initial_states", "pseudotime"]].groupby("initial_states").mean())

    file = f"pseudotime_{pCode}.tsv"
    mudata.obs["pseudotime"].to_csv(os.path.join(output_dir, file),
                                    index = True, header = True, sep="\t")

    metrics = _compute_results(mudata = mudata,
                                    code = code,
                                    seed = seed)
    _save_metrics(metrics, result_path)

    print("METRICS")
    print(metrics)

    _get_plots(mudata = mudata,
            code=code,
            seed = seed)

    # To correctly store the mudata the objects in .obsm require columns to be strings.
    mudata.obsm["DM_EigenVectors"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors"]]
    mudata.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors_multiscaled"]]

    mudata.write(os.path.join(output_dir,
                            f"{code}.h5mu")
                )


