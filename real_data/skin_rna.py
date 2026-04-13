import os
import atlas
import scipy
import pandas as pd
import numpy as np 
import muon as mu
import palantir
import cellrank
from muon import MuData
from anndata import AnnData
from .utils import _compute_results, _get_plots

def _invert_assignment(assignment):
    """Utility function for creating terminal/initial states from CellRank output"""
    if not isinstance(assignment.dtype, pd.CategoricalDtype):
        assignment = assignment.astype("category")

    inverted_assignment= { state: assignment.index[assignment==state].tolist()
                                for state in assignment.cat.categories}
    return inverted_assignment


def  _create_ts_dict(data:AnnData, fate_prob_key:str, cluster_key: str) -> dict:
    """Function creating terminal and initial states dicts from fate probabilities"""
    if fate_prob_key not in data.obsm or data.obsm[fate_prob_key] is None or data.obsm[fate_prob_key].shape[1] == 0:
        return {}
    terminal_cells = data.obsm[fate_prob_key].columns
    clusters = {}
    for cell in terminal_cells: 
        cluster = data.obs.loc[cell, cluster_key]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(cell)
    return clusters

def compute_entropy(data: MuData, fate_prob_key:str="palantir_fate_probabilities"):
    """Function mimicking ATLAS entropy computation"""
    def _minmax(x:np.ndarray) -> np.ndarray:
        if np.max(x) == np.min(x):
            return np.zeros_like(x)
        return (x- np.min(x)) / (np.max(x) - np.min(x))

    probs = data.obsm.get(fate_prob_key, None)
    if probs is None:
        raise ValueError("Compute fate probabilities before running entropy")

    if not isinstance(probs, pd.DataFrame):
        raise ValueError("Fate probabilities not a DataFrame")

    if (probs.shape[1] == 0
        or np.any(probs.sum(axis=1) == 0)
        or np.any(probs.sum(axis=0) == 0)):
        warnings.warn("No terminal states or cells with no developmental probability or state without assignment")
        data.obs["shannon_entropy"] = np.nan
        data.obs["kl_divergence"] = np.nan
        return

    shannon_entropy = scipy.stats.entropy(probs, axis=1)
    average_distribution = np.mean(probs, axis=0)
    kl_divergence = np.nan_to_num(scipy.stats.entropy(probs, average_distribution, axis=1, base=2),
                            nan=1.0,
                            copy=False)
    shannon_entropy, kl_divergence = _minmax(shannon_entropy), _minmax(kl_divergence)
    data.obs["shannon_entropy"] = pd.Series(shannon_entropy, index = data.obs.index)
    data.obs["kl_divergence"] = pd.Series(kl_divergence, index = data.obs.index)



if __name__ == "__main__":
    seed = 42
    np.random.seed(seed) 
    
    output_path = os.path.join("output", "mouse_skin")
    multi_path = os.path.join(output_path, "20:10_15:15:None_hard.h5mu")

    data = mu.read_h5mu(multi_path)

    # Adjust not to overwrite data
    data.obs["cellrank_she"] = data.obs["shannon_entropy"]
    data.obs["cellrank_kl"] = data.obs["kl_divergence"]
    data.obsm["fate_probabilities_cellrank"] = data.obsm["fate_probabilities"]
    data.uns["terminal_states_cellrank"] = data.uns["terminal_states"]
    data.uns["intermediate_states_cellrank"] = data.uns["intermediate_states"]
    data.uns["initial_states_cellrank"] = data.uns["initial_states"]

    # Select initial cell for Palantir computation to be the same aused in ATLAS
    early_cell = data.uns["palantir_initial_states"]["TAC-1"][0]

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
    clusters =  rna.obs.loc[rna.obsm["rna_fates_palantir"].columns, "celltype"].astype(str)
    rna.obsm["rna_fates_palantir"].columns = clusters
    rna.obsm["rna_fates_palantir"] = rna.obsm["rna_fates_palantir"].groupby(level=0, axis=1).sum()
    data.obsm["rna_fates_palantir"] = rna.obsm["rna_fates_palantir"]
    data.obs["rna_pseudotime"] = rna.obs["rna_pseudotime"].copy().reindex(data.obs.index)

    compute_entropy(data = data, fate_prob_key="rna_fates_palantir")

    # Identify terminal states
    terminal_rna = []
    terminal_atlas = []
    for k, v in data.uns["terminal_states"].items():
        terminal_rna.extend(v)
    for k, v in data.uns["palantir_terminal_states"].items():
        terminal_atlas.extend(v)

    data.obs["is_palantir_rna"] = (data.obs_names.isin(terminal_rna)) | (data.obs_names == early_cell)
    data.obs["is_palantir_atlas"] = (data.obs_names.isin(terminal_atlas)) | (data.obs_names == early_cell)
    mu.pl.embedding(data, basis="X_umap", color=["is_palantir_rna", "is_palantir_atlas"], save = "TI_atlas_vs_rna_palantir.png")

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

    # CELLRANK 
    rna.obs["celltype"] = rna.obs["celltype"].astype("category")
    kernel = cellrank.kernels.PseudotimeKernel(rna, 
                                                time_key = "rna_pseudotime",
                                                connectivity_key = "connectivities")
    kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = -1)
    g = cellrank.estimators.GPCCA(kernel)
    g.compute_schur()
    g.compute_macrostates(n_states=None, cluster_key = "celltype")
    g.predict_terminal_states(allow_overlap=True)
    g.predict_initial_states(allow_overlap=True)
    g.compute_fate_probabilities(use_petsc=True, n_jobs=-1)
    data.obsm["rna_fates_cellrank"] = pd.DataFrame(g.fate_probabilities.X,
                                                    index = rna.obs_names,
                                                    columns = g.fate_probabilities.names)
    data.uns["initial_states"] = _invert_assignment(g.initial_states)
    data.uns["terminal_states"] = _invert_assignment(g.terminal_states)
    compute_entropy(data = data, fate_prob_key="rna_fates_cellrank")

    terminal_rna = []
    terminal_atlas = []
    for k, v in data.uns["terminal_states"].items():
        terminal_rna.extend(v)
    for k, v in data.uns["palantir_terminal_states"].items():
        terminal_atlas.extend(v)

    data.obs["is_palantir_rna"] = (data.obs_names.isin(terminal_rna)) | (data.obs_names == early_cell)
    data.obs["is_palantir_atlas"] = (data.obs_names.isin(terminal_atlas)) | (data.obs_names == early_cell)
    mu.pl.embedding(data, basis="X_umap", color=["is_palantir_rna", "is_palantir_atlas"], save = "TI_atlas_vs_rna_pseudotimeK.png")

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










