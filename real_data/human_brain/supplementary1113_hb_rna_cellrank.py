########################################################################
# This code applies TI strategies on the Human Fetal Brain data.       #
# It presents scRNA-seq only based results                             # 
# Plots can be visualized in Supp. Figure 11 Supp Figure 13            #
########################################################################

import os
import atlas
import argparse
import pandas as pd
import numpy as np 
import muon as mu
import cellrank
from muon import MuData
from atlas.tl.utils import compute_entropy, _assign_state_colors, _resolve_overlap, reset_state_colors
from real_data.utils import _compute_results, _get_plots, _save_metrics



if __name__ == "__main__":
    seed = 42
    np.random.seed(seed) 
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_rna", type=int, default=15)
    parser.add_argument("--knn_act", type=int, default=15)
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--wnn", default=None)
    parser.add_argument("--n_states", type=int, default=4)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--removed", type=int, default=0)
    args=parser.parse_args()

    removed = bool(args.removed)
    prefix = "removed" if removed else "non_removed"
    pCode = f"{prefix}_{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}_{args.n_states}"
    code = f"rna_cellrank_{pCode}"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "human_brain")
    result_path = os.path.join(output_dir, "results.tsv")

    mudata = mu.read_h5mu(os.path.join(output_dir,
                        f"cellrank_{pCode}.h5mu")
            )

    file = f"rna_pseudotime_{prefix}_{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}.tsv"
    pseudotime = pd.read_csv(os.path.join(output_dir, file),
                            sep = "\t", header=0, index_col=0)

    # CELLRANK 
    rna = mudata["rna"]
    rna.obs["Cluster.Name"] = mudata.obs["Cluster.Name"].reindex(rna.obs.index)
    rna.obs["pseudotime"] = pseudotime.reindex(rna.obs.index)

    kernel = cellrank.kernels.PseudotimeKernel(rna, 
                                                time_key = "pseudotime",
                                                connectivity_key = "connectivities")
    kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = args.n_jobs)
    g = cellrank.estimators.GPCCA(kernel)
    g.compute_schur()
    g.compute_macrostates(n_states=args.n_states, cluster_key = "Cluster.Name")

    try:
        g.predict_terminal_states(allow_overlap=False)
        g.predict_initial_states(allow_overlap=False)
        allow_overlap = False
    except Exception as e:
        print(f"allow_overlap = False gave Exception \n{e}\n. Trying with allow_overlap = True")
        g.predict_terminal_states(allow_overlap=True)
        g.predict_initial_states(allow_overlap=True)
        allow_overlap = True

    rna.uns["cellrank"] = {"allow_overlap":allow_overlap}

    g.compute_fate_probabilities(use_petsc=True, n_jobs=args.n_jobs)

    rna.obsm["fate_probabilities"] = pd.DataFrame(g.fate_probabilities.X,
                                           index = rna.obs_names,
                                           columns = g.fate_probabilities.names)
    mudata.obsm["rna_fates"] = rna.obsm["fate_probabilities"]
    mudata.obs["rna_pseudotime"] = rna.obs["pseudotime"].reindex(mudata.obs.index)

    # ATLAS compute entropy overwrites the ATLAS entropy measures
    atlas_kl, atlas_she = mudata.obs["kl_divergence"].copy(), mudata.obs["shannon_entropy"].copy()
    compute_entropy(mudata = mudata,
            fate_probability_key = "rna_fates")

    # _assign_state_colors overwrites the ATLAS colors
    obs_is, obs_ms, obs_ts = mudata.obs["initial_states"].copy(), mudata.obs["macrostates"].copy(), mudata.obs["terminal_states"].copy() 
    atlas_colors = mudata.uns["atlas_state_palette"].copy()
    del mudata.uns["atlas_state_palette"]

    mudata.obs["initial_states"] = g.initial_states.reindex(mudata.obs.index)
    mudata.obs["terminal_states"] = g.terminal_states.reindex(mudata.obs.index)
    mudata.obs["macrostates"] = g.macrostates.reindex(mudata.obs.index)
    _assign_state_colors(mudata)

    print("AVERAGE PSEUDOTIME PER MACROSTATE:")
    print(mudata.obs[["macrostates", "rna_pseudotime"]].groupby("macrostates").mean())

    # Compute metrics
    metrics = _compute_results(mudata = mudata,
                        code = code,
                        seed = seed, 
                        time_key = "rna_pseudotime",
                        fate_key = "rna_fates")
    _save_metrics(metrics, result_path)
    print("METRICS:")
    print(metrics)

    # Plots
    _get_plots(mudata = mudata,
            code = code,
            cluster_key = "Cluster.Name",
            time_key = "rna_pseudotime",
            embedding_key = "umap",
            fate_key = "rna_fates",
            seed = seed)

    # Update the muon object and save withour redundancies 
    mudata.obs["rna_kl_divergence"] = mudata.obs["kl_divergence"]
    mudata.obs["rna_shannon_entropy"] = mudata.obs["shannon_entropy"]
    mudata.obs["kl_divergence"] = atlas_kl
    mudata.obs["shannon_entropy"] = atlas_she
    del mudata.obsm["rna_fates"]
    mudata.obs["rna_terminal_states"] = mudata.obs["terminal_states"]
    mudata.obs["rna_initial_states"] = mudata.obs["initial_states"]
    mudata.obs["rna_macrostates"] = mudata.obs["macrostates"]
    mudata.obs["macrostates"] = obs_ms
    mudata.obs["terminal_states"] = obs_ts
    mudata.obs["initial_statest"] = obs_is
    mudata.uns["rna_state_palette"] = mudata.uns["atlas_state_palette"] 
    mudata.uns["atlas_state_palette"] = atlas_colors
    mudata.uns["rna_terminal_states_colors"] = mudata.uns["terminal_states_colors"]
    mudata.uns["rna_initial_states_colors"] = mudata.uns["initial_states_colors"]
    mudata.uns["rna_macrostates_colors"] = mudata.uns["macrostates_colors"]
    reset_state_colors(mudata)

    mudata.write(os.path.join(output_dir,
                                f"{code}.h5mu"))




