import os
import json
import time
import atlas 
import numpy as np
import pandas as pd
import muon as mu
import argparse
import fcntl
from .utils import _compute_results, _get_plots, _compute_enrichment
from atlas.tl import PalantirExtension, CellRankExtension


_placeholder = {"code": None,
                "pearson_stat_KL": None,
                "pearson_pval_KL": None,
                "pearson_ciL_KL": None,
                "pearson_ciH_KL": None,
                "pearson_stat_SHE": None,
                "pearson_pval_SHE": None,
                "pearson_ciL_SHE": None,
                "pearson_ciH_SHE": None,
                "spearman_stat_KL": None,
                "spearman_pval_KL": None,
                "spearman_ciL_KL": None,
                "spearman_ciH_KL": None,
                "spearman_stat_SHE": None,
                "spearman_pval_SHE": None,
                "spearman_ciL_SHE": None,
                "spearman_ciH_SHE": None,
                "fate_index_stat": None,
                "fate_index_pval": None,
                "fate_index_ciL": None,
                "fate_index_ciH": None,
                "pseudotime_enrichment": None,
                "silhouette_soft": None,
                "silhouette_hard": None,
                "n_terminal_states": None,
                "n_intermediate_states": None,
                "allow_overlap": None,
                "intermediate_enrichment": None,
                "initial_enrichment": None,
                "strategy": None,
                "failed": True
           }



if __name__=="__main__":
    seed, threads = 42, 3
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=int)
    parser.add_argument("--activity", type=int)
    parser.add_argument("--wnn", type = int, default = None)
    parser.add_argument("--states", type = int, default = None)
    args = parser.parse_args()

    knn_rna, knn_act, wnn = args.rna, args.activity, args.wnn
    n_pcs_rna, n_pcs_act= 20, 10
    code = f"{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard"
    _placeholder["code"] = code
    n_states = args.states

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "mouse_skin")
    results_path = os.path.join(output_dir, "results.csv")
    data_path = os.path.join(output_dir, "skin.h5mu")

    mudata = mu.read_h5mu(data_path)
    # select palantir initial cell
    initial = rng.choice(mudata[mudata.obs["celltype"]=="TAC-1"].obs_names)

    new_data = atlas.pp.preprocessing(mudata = mudata,
                n_pcs_rna = n_pcs_rna, 
                n_pcs_act = n_pcs_act,
                knn_rna = knn_rna,
                knn_act = knn_act,
                n_neighbors = wnn,
                features = None)

    failed = False
    try:
        pex = PalantirExtension(mudata = new_data)
        pex.compute_kernel()
        pex.compute_diffusion_maps(seed = seed)
        pex.compute_multiscale_space()

        pex.run(early_cell = initial,
                    cluster_key = "celltype",
                    pseudotime_key = "pseudotime",
                    fate_prob_key = "palantir_probabilities",
                    terminal_states = None,
                    n_jobs = threads,
                    random_state = seed)
    except Exception:
        failed = True

    if failed:
        palantirM = _placeholder
    else:
        palantirM = _compute_results(mudata = new_data,
                                     code = code,
                                     seed = seed,
                                     time_key = "pseudotime",
                                     fate_key = "palantir_probabilities")

        palantirM["n_terminal_states"] = len(new_data.uns["terminal_states"])
        palantirM["n_intermediate_states"] = np.nan
        palantirM["allow_overlap"] = np.nan
        palantirM["intermediate_enrichment"] = np.nan
        palantirM["initial_enrichment"] = np.nan
        palantirM["strategy"] = "palantir"
        palantirM["failed"] = failed

    with open(results_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pd.DataFrame([palantirM]).to_csv(f, index=False, header=f.tell() == 0)
        fcntl.flock(f, fcntl.LOCK_UN)

    failed = False
    try:
        cex = CellRankExtension(mudata=new_data)
        cex.compute_kernel(connectivity_key = "wnn_connectivities",
                        time_key = "pseudotime",
                        cluster_key = "celltype",
                        n_jobs = threads)
    except Exception:
        failed = True

    try: 
        cex.run(n_states = n_states, 
            use_petsc = True,
            allow_overlap = False,
            n_jobs = threads)
        allow_overlap = False
    except Exception:
        try:
            cex.run(n_states = n_states, 
                use_petsc = True,
                allow_overlap = True,
                n_jobs = threads)
            allow_overlap=True
        except Exception:
            failed = True

    if failed: 
        pseudotimeK = _placeholder
    else:
        pseudotimeK = _compute_results(mudata = new_data,
                                    code = code,
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "fate_probabilities")

        pseudotimeK["n_terminal_states"] = len(new_data.uns["terminal_states"])
        pseudotimeK["n_intermediate_states"] = len(new_data.uns["intermediate_states"])
        pseudotimeK["allow_overlap"] = allow_overlap 

        pseudotime = new_data.obs["pseudotime"]
        initial_key = list(new_data.uns["initial_states"].keys())[0]
        initial_cells = new_data.uns["initial_states"][initial_key]
        pseudotimeK["initial_enrichment"] = _compute_enrichment(pseudotime, initial_cells)
    
        if pseudotimeK["n_intermediate_states"] > 0:
            intermediate_series = pd.Series(index = new_data.uns["intermediate_states"].keys(), dtype = float)
            for k, v in new_data.uns["intermediate_states"].items():
                intermediate_series[k] = _compute_enrichment(pseudotime, v)
            intermediate_enrichment = intermediate_series.mean()
        else:
            intermediate_enrichment = np.nan
        pseudotimeK["intermediate_enrichment"] = intermediate_enrichment
        pseudotimeK["strategy"] = "pseudotime-kernel"

    with open(results_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pd.DataFrame([pseudotimeK]).to_csv(f, index=False, header=f.tell() == 0)
        fcntl.flock(f, fcntl.LOCK_UN)

    



