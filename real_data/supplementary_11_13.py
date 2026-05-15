import os
import atlas
import argparse
import pandas as pd
import numpy as np 
import muon as mu
import palantir
import cellrank
from muon import MuData
from .utils import _compute_results, _get_plots, _assign_state_colors, _invert_assignment, _create_ts_dict, compute_entropy



if __name__ == "__main__":
    seed = 42
    np.random.seed(seed) 
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--removed", type=int, default=1)
    parser.add_argument("--n_states", type=int, default=5)
    args = parser.parse_args()
    n_states, removed = args.n_states, bool(args.removed)
    prefix = "brainR" if removed else "brain"

    output_path = os.path.join("output", "human_brain")
    multi_path = os.path.join(output_path, f"{prefix}_{n_states}.h5mu")

    data = mu.read_h5mu(multi_path)

    # Reset states slots from ATLAS run
    reset_states = ["initial_states", "intermediate_states", "terminal_states"]
    for state in reset_states:
        if state in data.uns.keys():
            data.uns[state] = None

    # Select initial cell for Palantir computation to be the same aused in ATLAS
    early_cell = data.uns["palantir_initial_states"]["Cyc. Prog."][0]

    rna = data["rna"]
    rna.obs["Cluster.Name"] = data.obs["Cluster.Name"].copy().reindex(rna.obs.index)

    # PALANTIR ON RNA DATA
    kernel = palantir.utils.compute_kernel(rna, knn=30)
    diffusion_maps = palantir.utils.diffusion_maps_from_kernel(kernel = kernel, seed = seed)
    rna.obsp["DM_Similarity"] = diffusion_maps["T"]
    rna.obsm["DM_EigenVectors"] = diffusion_maps["EigenVectors"].set_index(rna.obs.index)
    rna.uns["DM_EigenValues"] = diffusion_maps["EigenValues"].values

    eigenvectors = pd.DataFrame(rna.obsm["DM_EigenVectors"], index = rna.osb_names) if not isinstance(rna.obsm["DM_EigenVectors"], pd.DataFrame) else rna.obsm["DM_EigenVectors"]
    dm_dict = {"EigenValues": rna.uns["DM_EigenValues"], "EigenVectors": eigenvectors}
    result = palantir.utils.determine_multiscale_space(dm_res = dm_dict)
    rna.obsm["DM_EigenVectors_multiscaled"] = result
    palantir.core.run_palantir(data=rna,
                        early_cell = early_cell,
                        pseudo_time_key = "rna_pseudotime",
                        seed = seed,
                        fate_prob_key = "rna_fates_palantir")

    data.uns["terminal_states"] = _create_ts_dict(rna, fate_prob_key="rna_fates_palantir", cluster_key="Cluster.Name")
    data.uns["initial_states"] = {rna.obs.loc[early_cell, "Cluster.Name"]: early_cell}
    _assign_state_colors(data)
    clusters =  rna.obs.loc[rna.obsm["rna_fates_palantir"].columns, "Cluster.Name"].astype(str)
    rna.obsm["rna_fates_palantir"].columns = clusters
    rna.obsm["rna_fates_palantir"] = rna.obsm["rna_fates_palantir"].groupby(level=0, axis=1).sum()
    data.obsm["rna_fates_palantir"] = rna.obsm["rna_fates_palantir"]
    data.obs["rna_pseudotime"] = rna.obs["rna_pseudotime"].copy().reindex(data.obs.index)

    compute_entropy(data = data, fate_prob_key="rna_fates_palantir")

    # Identify terminal states
    terminal_rna = []
    for k, v in data.uns["terminal_states"].items():
        terminal_rna.extend(v)
    print(data.obs.loc[terminal_rna, "rna_pseudotime"])

    data.obs["is_TI_rna"] = (data.obs_names.isin(terminal_rna))
    mu.pl.embedding(data, basis="X_umap", color=["is_TI_rna"], save = "HF_palantir_selected_rna.png")

    # Compute metrics
    metrics = _compute_results(mudata = data,
                        code = "rna",
                        seed = seed, 
                        time_key = "rna_pseudotime",
                        fate_key = "rna_fates_palantir")
    print(metrics)
    avg_pseudotime = data.obs[["Cluster.Name", "rna_pseudotime"]].groupby("Cluster.Name").mean()
    print(avg_pseudotime)
    
    # Plots
    _get_plots(mudata = data,
            code = "rna",
            time_key = "rna_pseudotime",
            embedding_key = "umap",
            fate_key = "rna_fates_palantir",
            seed = seed,
            ti_strategy = "palantir")

    data.uns["rna_palantir_initial"] = data.uns["initial_states"]
    data.uns["rna_palantir_terminal"] = data.uns["terminal_states"]
    data.uns["rna_palantir_colors"] = data.uns["fate_state_colors"]

    # CELLRANK 
    rna.obs["Cluster.Name"] = rna.obs["Cluster.Name"].astype("category")
    kernel = cellrank.kernels.PseudotimeKernel(rna, 
                                                time_key = "rna_pseudotime",
                                                connectivity_key = "connectivities")
    kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = -1)
    g = cellrank.estimators.GPCCA(kernel)
    g.compute_schur()
    g.compute_macrostates(n_states=n_states, cluster_key = "Cluster.Name")
    g.predict_terminal_states(allow_overlap=True)
    g.predict_initial_states(allow_overlap=True)
    g.compute_fate_probabilities(use_petsc=True, n_jobs=-1)
    data.obsm["rna_fates_cellrank"] = pd.DataFrame(g.fate_probabilities.X,
                                                    index = rna.obs_names,
                                                    columns = g.fate_probabilities.names)
    data.uns["initial_states"] = _invert_assignment(g.initial_states)
    data.uns["terminal_states"] = _invert_assignment(g.terminal_states)
    data.uns["all_states"] = _invert_assignment(g.macrostates)
    _assign_state_colors(data)
    compute_entropy(data = data, fate_prob_key="rna_fates_cellrank")


    terminal = data.uns["terminal_states"] 
    initial = data.uns["initial_states"]
    intermediate = data.uns["intermediate_states"]
    intermediate = {} if intermediate is None else intermediate

    avg = {k: data.obs.loc[cells, "rna_pseudotime"].mean() for k, cells in terminal.items()}
    print("terminal", avg)
    avg = {k: data.obs.loc[cells, "rna_pseudotime"].mean() for k, cells in initial.items()}
    print("terminal", avg)
    if len(intermediate) > 0:
        avg = {k: data.obs.loc[cells, "rna_pseudotime"].mean() for k, cells in intermediate.items()}
        print("intermediate", avg)
    else:
        print("NO INTERMEDIATE CELLS DETECTED")

    data.obs["is_TI_rna"] = "other"
    for k, cells in initial.items():
        data.obs.loc[cells, "is_TI_rna"] = "initial"
    for k, cells in intermediate.items():
        data.obs.loc[cells, "is_TI_rna"] = "intermediate"
    mu.pl.embedding(data, basis="X_umap", color=["is_TI_rna"], palette = {"other": "lightgrey", "initial": "red", "intermediate": "blue"}, save = "HF_pseudotimeK_selected_rna.png")

    # Compute metrics
    metrics = _compute_results(mudata = data,
                        code = "rna",
                        seed = seed, 
                        time_key = "rna_pseudotime",
                        fate_key = "rna_fates_cellrank")
    print(metrics)
    
    # Plots
    _get_plots(mudata = data,
            code = "rna",
            time_key = "rna_pseudotime",
            embedding_key = "umap",
            fate_key = "rna_fates_cellrank",
            seed = seed,
            ti_strategy = "pseudotime-kernel")
    
    data["rna"].obsm["DM_EigenVectors"].columns = [str(c) for c in data["rna"].obsm["DM_EigenVectors"].columns]
    data["rna"].obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in data["rna"].obsm["DM_EigenVectors_multiscaled"].columns]

    data.write_h5mu(os.path.join(output_path, f"rna_{prefix}_{n_states}.h5mu"))












