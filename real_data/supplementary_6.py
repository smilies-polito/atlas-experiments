import os
import json
import time
import fcntl
import atlas 
import argparse
import tracemalloc
import numpy as np
import pandas as pd
import muon as mu
from .utils import _compute_results, _get_plots
from atlas.tl import PalantirExtension, CellRankExtension


if __name__=="__main__":
    seed = 42
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_states", type=int, default=6)
    args = parser.parse_args()
    n_states = args.n_states
    allow_overlap = True if n_states == 6 else False

    #args 
    knn_rna, knn_act, wnn = 15, 15, None
    n_pcs_rna, n_pcs_act= 20, 10
    code = f"{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "embryonic_mouse_brain")
    data_path = os.path.join(output_dir, "emb.h5mu")
    features_path = os.path.join(output_dir, "features.tsv")

    mudata = mu.read_h5mu(data_path)

    # select palantir initial cell
    initial = rng.choice(mudata[mudata.obs["celltype"]=="RG, Astro, OPC"].obs_names)
    features = pd.read_csv(features_path, sep = "\t", header=0, index_col=0)

    new_data = atlas.pp.preprocessing(mudata = mudata,
                n_pcs_rna = n_pcs_rna, 
                n_pcs_act = n_pcs_act,
                knn_rna = knn_rna,
                knn_act = knn_act,
                n_neighbors = wnn,
                features = features)

    mu.tl.louvain(new_data)
    mu.pl.embedding(new_data, basis = "X_umap", color=["louvain", "celltype"], save =f"{code}_cluster.png")

    # TRAJECTORY INFERENCE USING PALANTIR
    pex = PalantirExtension(mudata = new_data)
    pex.compute_kernel()
    pex.compute_diffusion_maps(seed = seed)
    pex.compute_multiscale_space()

    pex.run(early_cell = initial,
                cluster_key = "celltype",
                pseudotime_key = "pseudotime",
                fate_prob_key = "palantir_probabilities",
                terminal_states = None,
                n_jobs = -1,
                random_state = seed)
    avg_pseudotime = new_data.obs[["pseudotime", "celltype"]].groupby("celltype").mean()
    print(avg_pseudotime)

    palantir_metrics = _compute_results(mudata = new_data,
                                    code = code,
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "palantir_probabilities")
    print(palantir_metrics)
    terminal_states = []
    for t, v in new_data.uns["terminal_states"].items():
        terminal_states.extend(v)

    new_data.obs["is_TI"] = new_data.obs_names.isin(terminal_states)
    mu.pl.embedding(new_data, basis="X_umap", color = ["is_TI"], save = "e18_palantir_selected_states.png")
 
    _get_plots(mudata = new_data,
            code = code,
            time_key = "pseudotime",
            fate_key = "palantir_probabilities",
            seed = seed,
            ti_strategy = "palantir")

    # rename not to overwrite + adjust colnames for saving purposes 
    new_data.obs["palantir_SHE"] = new_data.obs["shannon_entropy"]
    new_data.obs["palantir_KL"] = new_data.obs["kl_divergence"]
    new_data.uns["palantir_terminal_states"] = new_data.uns["terminal_states"]
    new_data.uns["palantir_initial_states"] = new_data.uns["initial_states"]
    new_data.uns["palantir_fate_colors"] = new_data.uns["fate_state_colors"] 
    new_data.obsm["DM_EigenVectors"].columns = [str(c) for c in new_data.obsm["DM_EigenVectors"].columns]
    new_data.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in new_data.obsm["DM_EigenVectors_multiscaled"].columns]

    cex = CellRankExtension(mudata=new_data)
    cex.compute_kernel(connectivity_key = "wnn_connectivities",
                    time_key = "pseudotime",
                    cluster_key = "celltype",
                    n_jobs = -1)
    cex.run(n_states = n_states,
        use_petsc = True,
        allow_overlap = allow_overlap,
        n_jobs = -1)

    pseudotimeK_metrics = _compute_results(mudata = new_data,
                                    code = code,
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "fate_probabilities")
    print(pseudotimeK_metrics)

    _get_plots(mudata = new_data,
            code = code,
            time_key = "pseudotime",
            fate_key = "fate_probabilities",
            seed = seed,
            ti_strategy = "pseudotimeK")

    avg_pseudotime = { k: new_data.obs.loc[cells, "pseudotime"].mean() for k, cells in new_data.uns["terminal_states"].items()}
    print("terminal", avg_pseudotime)
    avg_pseudotime = { k: new_data.obs.loc[cells, "pseudotime"].mean() for k, cells in new_data.uns["intermediate_states"].items()}
    print("intermediate", avg_pseudotime)
    avg_pseudotime = { k: new_data.obs.loc[cells, "pseudotime"].mean() for k, cells in new_data.uns["initial_states"].items()}
    print("initial", avg_pseudotime)

    new_data.obs["inferred"] = "other"
    for k,v in new_data.uns["initial_states"].items():
        new_data.obs.loc[v, "inferred"] = "initial"
    for k,v in new_data.uns["intermediate_states"].items():
        new_data.obs.loc[v, "inferred"] = "intermediate"
    mu.pl.embedding(new_data, basis="X_umap", color = ["inferred"], palette = {"initial":"red", "intermediate": "blue", "other":"grey"}, save = "e18_pseudotimeK_selected_states.png")

    new_data.obs["pseudotimeK_SHE"] = new_data.obs["shannon_entropy"]
    new_data.obs["pseudotimeK_KL"] = new_data.obs["kl_divergence"]
    new_data.uns["pseudotimeK_terminal_states"] = new_data.uns["terminal_states"]
    new_data.uns["pseudotimeK_initial_states"] = new_data.uns["initial_states"]
    if "intermediate_states" in new_data.uns:
        new_data.uns["pseudotimeK_intermediate_states"] = new_data.uns["intermediate_states"]
    new_data.uns["pseudotimeK_fate_colors"] = new_data.uns["fate_state_colors"] 
    new_data.obsm["pseudotimeK_fate_probabilities"] =new_data.obsm["fate_probabilities"]


    new_data.write(os.path.join(output_dir, f"{code}_{n_states}states.h5mu"))

