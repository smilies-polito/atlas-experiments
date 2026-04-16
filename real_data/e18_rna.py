import os
import atlas
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
    
    output_path = os.path.join("output", "embryonic_mouse_brain")
    multi_path = os.path.join(output_path, "20:10_15:15:None_hard.h5mu")
    n_states = 6

    data = mu.read_h5mu(multi_path)


    # Select initial cell for Palantir computation to be the same aused in ATLAS
    early_cell = data.uns["palantir_initial_states"]["RG, Astro, OPC"][0]

    rna = data["rna"]
    rna.obs["celltype"] = data.obs["celltype"].copy().reindex(rna.obs.index)

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

    data.uns["terminal_states"] = _create_ts_dict(rna, fate_prob_key="rna_fates_palantir", cluster_key="celltype")
    data.uns["initial_states"] = {rna.obs.loc[early_cell, "celltype"]: early_cell}
    _assign_state_colors(data)
    clusters =  rna.obs.loc[rna.obsm["rna_fates_palantir"].columns, "celltype"].astype(str)
    rna.obsm["rna_fates_palantir"].columns = clusters
    rna.obsm["rna_fates_palantir"] = rna.obsm["rna_fates_palantir"].groupby(level=0, axis=1).sum()
    data.obsm["rna_fates_palantir"] = rna.obsm["rna_fates_palantir"]
    data.obs["rna_pseudotime"] = rna.obs["rna_pseudotime"].copy().reindex(data.obs.index)

    compute_entropy(data = data, fate_prob_key="rna_fates_palantir")

    # Identify terminal states
    terminal_rna = []
    for k, v in data.uns["terminal_states"].items():
        terminal_rna.extend(v)
    for k, v in data.uns["initial_states"].items():
        terminal_rna.extend(v)

    data.obs["is_TI"] = (data.obs_names.isin(terminal_rna))
    mu.pl.embedding(data, basis="X_umap", color=["is_TI"], save = "e18_palantir_selected_rna.png")

    # Compute metrics
    metrics = _compute_results(mudata = data,
                        code = "rna",
                        seed = seed, 
                        time_key = "rna_pseudotime",
                        fate_key = "rna_fates_palantir")
    print(metrics)
    
    # Plots
    _get_plots(mudata = data,
            code = "rna",
            time_key = "rna_pseudotime",
            embedding_key = "umap",
            fate_key = "rna_fates_palantir",
            seed = seed,
            ti_strategy = "palantir")

    avg_pseudotime = data.obs[["celltype", "rna_pseudotime"]].groupby("celltype").mean()
    print(avg_pseudotime)


    # CELLRANK 
    rna.obs["celltype"] = rna.obs["celltype"].astype("category")
    kernel = cellrank.kernels.PseudotimeKernel(rna, 
                                                time_key = "rna_pseudotime",
                                                connectivity_key = "connectivities")
    kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = -1)
    g = cellrank.estimators.GPCCA(kernel)
    g.compute_schur()
    g.compute_macrostates(n_states=n_states, cluster_key = "celltype")
    g.predict_terminal_states(allow_overlap=True)
    g.predict_initial_states(allow_overlap=True)
    g.compute_fate_probabilities(use_petsc=True, n_jobs=-1)
    data.obsm["rna_fates_cellrank"] = pd.DataFrame(g.fate_probabilities.X,
                                                    index = rna.obs_names,
                                                    columns = g.fate_probabilities.names)
    data.uns["initial_states"] = _invert_assignment(g.initial_states)
    data.uns["terminal_states"] = _invert_assignment(g.terminal_states)
    _assign_state_colors(data)
    compute_entropy(data = data, fate_prob_key="rna_fates_cellrank")

    terminal_rna = []
    for k, v in data.uns["terminal_states"].items():
        terminal_rna.extend(v)
    for k, v in data.uns["initial_states"].items():
        terminal_rna.extend(v)

    data.obs["is_TI"] = (data.obs_names.isin(terminal_rna))
    mu.pl.embedding(data, basis="X_umap", color=["is_TI"], save = "e18_pseudotimeK_selected_rna.png")

    d = data.uns["terminal_states"]
    print(d)
    cell_to_label = {cell: label for label, cells in d.items() for cell in cells}
    data.obs["is_TI"] = data.obs_names.map(cell_to_label)
    macrostate_composition_T = data.obs[["celltype", "is_TI"]].groupby(["celltype", "is_TI"]).size().to_frame(name="count")

    d = data.uns["initial_states"]
    cell_to_label = {cell: label for label, cells in d.items() for cell in cells}
    data.obs["is_TI"] = data.obs_names.map(cell_to_label)
    macrostate_composition_I = data.obs[["celltype", "is_TI"]].groupby(["celltype", "is_TI"]).size()
    print(macrostate_composition_I)

    # Compute metrics
    metrics = _compute_results(mudata = data,
                        code = "rna",
                        seed = seed, 
                        time_key = "rna_pseudotime",
                        fate_key = "rna_fates_cellrank")
    
    # Plots
    _get_plots(mudata = data,
            code = "rna",
            time_key = "rna_pseudotime",
            embedding_key = "umap",
            fate_key = "rna_fates_cellrank",
            seed = seed,
            ti_strategy = "pseudotime-kernel")
    
    print(metrics)
    macrostate_composition_T.to_csv(os.path.join(output_path, "macrostate_rna.tsv"), sep="\t", header = True, index = True)











