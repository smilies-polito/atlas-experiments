import os
import json
import time
import atlas 
import argparse
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
    parser.add_argument("--removed", type=int, default=1)
    parser.add_argument("--n_states", type=int, default = 5)
    args = parser.parse_args()
    removed = bool(args.removed)
    prefix = "brainR" if removed else "brain"
    n_states = args.n_states

    if removed:
        allow_overlap = True if n_states == 5 else False
    else:
        allow_overlap = True

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "human_brain")
    data_path = os.path.join(output_dir, f"{prefix}.h5mu")

    mudata = mu.read_h5mu(data_path)
    initial = rng.choice(mudata[mudata.obs["Cluster.Name"]=="Cyc. Prog."].obs_names)

    pex = PalantirExtension(mudata = mudata)
    pex.compute_kernel()
    pex.compute_diffusion_maps(seed = seed)
    pex.compute_multiscale_space()

    pex.run(early_cell = initial,
                cluster_key = "Cluster.Name",
                pseudotime_key = "pseudotime",
                fate_prob_key = "palantir_probabilities",
                terminal_states = None,
                n_jobs = -1,
                random_state = seed)

    palantir_metrics = _compute_results(mudata = mudata,
                                    code = f"brain_{n_states}",
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "palantir_probabilities")
    print(palantir_metrics)

    average_cell_pseudotime = mudata.obs[["pseudotime", "Cluster.Name"]].groupby("Cluster.Name").mean()
    print(average_cell_pseudotime)

    _get_plots(mudata = mudata,
            code = f"brain_{n_states}",
            time_key = "pseudotime",
            fate_key = "palantir_probabilities",
            seed = seed,
            ti_strategy = "palantir")

    terminal_states = []
    for t,v in mudata.uns["terminal_states"].items():
        terminal_states.extend(v)
    mudata.obs["is_state_palantir"] = mudata.obs_names.isin(terminal_states)
    mu.pl.embedding(mudata, basis="X_umap", color=["is_state_palantir"], save = "brain_palantir_selected_states.png")
    print(mudata.obs["pseudotime"].loc[terminal_states])

    # rename not to overwrite + adjust colnames for saving purposes 
    mudata.obs["palantir_SHE"] = mudata.obs["shannon_entropy"]
    mudata.obs["palantir_KL"] = mudata.obs["kl_divergence"]
    mudata.uns["palantir_terminal_states"] = mudata.uns["terminal_states"]
    mudata.uns["palantir_initial_states"] = mudata.uns["initial_states"]
    mudata.uns["palantir_fate_colors"] = mudata.uns["fate_state_colors"] 
    mudata.obsm["DM_EigenVectors"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors"].columns]
    mudata.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors_multiscaled"].columns]

    cex = CellRankExtension(mudata=mudata)
    cex.compute_kernel(connectivity_key = "wnn_connectivities",
                    time_key = "pseudotime",
                    cluster_key = "Cluster.Name",
                    n_jobs = -1)
    cex.run(n_states = n_states, 
        use_petsc = True,
        allow_overlap = allow_overlap,
        n_jobs = -1)

    pseudotimeK_metrics = _compute_results(mudata = mudata,
                                    code = f"brain_{n_states}",
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "fate_probabilities")
    print(pseudotimeK_metrics)

    _get_plots(mudata = mudata,
            code = f"brain_{n_states}",
            time_key = "pseudotime",
            fate_key = "fate_probabilities",
            seed = seed,
            ti_strategy = "pseudotimeK")


    avg_pseudotime = { k: mudata.obs.loc[cells, "pseudotime"].mean() for k, cells in mudata.uns["terminal_states"].items()}
    print("terminal", avg_pseudotime)
    avg_pseudotime = { k: mudata.obs.loc[cells, "pseudotime"].mean() for k, cells in mudata.uns["intermediate_states"].items()}
    print("intermediate", avg_pseudotime)
    avg_pseudotime = { k: mudata.obs.loc[cells, "pseudotime"].mean() for k, cells in mudata.uns["initial_states"].items()}
    print("initial", avg_pseudotime)

    mudata.obs["inferred"] = "other"
    for k,v in mudata.uns["initial_states"].items():
        mudata.obs.loc[v, "inferred"] = "initial"
    for k,v in mudata.uns["intermediate_states"].items():
        mudata.obs.loc[v, "inferred"] = "intermediate"

    mu.pl.embedding(mudata, basis="X_umap", color = "inferred", palette = {"initial": "red", "intermediate": "blue", "other": "grey"}, save = "HF_intermediate.png")
   
    mudata.obs["pseudotimeK_SHE"] = mudata.obs["shannon_entropy"]
    mudata.obs["pseudotimeK_KL"] = mudata.obs["kl_divergence"]
    mudata.uns["pseudotimeK_terminal_states"] = mudata.uns["terminal_states"]
    mudata.uns["pseudotimeK_initial_states"] = mudata.uns["initial_states"]
    if "intermediate_states" in mudata.uns:
        mudata.uns["pseudotimeK_intermediate_states"] = mudata.uns["intermediate_states"]

    mudata.uns["pseudotimeK_fate_colors"] = mudata.uns["fate_state_colors"]
    mudata.obsm["pseudotimeK_fate_probabilities"] = mudata.obsm["fate_probabilities"]
    
    mudata.write(os.path.join(output_dir, f"{prefix}_{n_states}.h5mu"))
