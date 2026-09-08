import os
import warnings
import argparse
import cellrank 
import muon as mu 
import numpy as np
import pandas as pd
from atlas.tl.utils import compute_entropy, _resolve_overlap
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation_rna, _apply_metrics

        
if __name__=="__main__":
    seed = 42
    working_directory = os.getcwd() # set path to repository 
    np.random.seed(seed)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tree", type=str)
    parser.add_argument("--rd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--n_pcs_rna", type=int, default = 20)
    args = parser.parse_args()

    # DATA CONTRUCTION 
    tree = args.tree
    diff_cif_fraction, cif_sigma = args.rd, args.sigma
    knn_rna, knn_act, wnn= args.knn_rna, 30, 30
    n_pcs_rna, n_pcs_act = args.n_pcs_rna, 10
    muCode = f"{tree}_{diff_cif_fraction}:{cif_sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}"
    code = f"{tree}_{diff_cif_fraction}:{cif_sigma}_{n_pcs_rna}_{knn_rna}"

    data_path = os.path.join(working_directory, "output", "simulations", "mudata")
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "pseudotime_kernel_rna")
    mudata = mu.read_h5mu(os.path.join(data_path, f"{muCode}.h5mu"))
    mudata.pull_obs()

    # ground truth construction
    early_cell = initial_macrostate(mudata = mudata, 
            pseudotime_key = "rna:pseudotime", 
            n_cells=30)
    early_cluster_subset = mudata.obs["rna:pop"].loc[early_cell].value_counts().idxmax()
    initial_cells = {early_cluster_subset: early_cell}

    terminal_cells = {}
    for branch in TERM_DICT[tree]:
        cell = terminal_macrostate(mudata = mudata,
                pseudotime_key = "rna:pseudotime", 
                cluster_key = "rna:pop",
                terminal_state = branch,
                n_cells=30)            
        terminal_cells[branch] = cell

    true_fates = truth_like_fates(
            pseudotime = mudata.obs["rna:pseudotime"],
            membership = mudata.obs["rna:pop"],
            tree = tree
            )

    rna = mudata["rna"].copy()
    try:
        kernel = cellrank.kernels.PseudotimeKernel(rna,
                                            time_key = "pseudotime",
                                            connectivity_key = "connectivities")
        kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = -1)
        failed = False
    except Exception as e:
        print(e)
        failed = True
    
    try:
        g = cellrank.estimators.GPCCA(kernel)
        g.compute_schur()
        g.compute_macrostates(n_states=None, cluster_key = "pop")
        g.predict_terminal_states(allow_overlap=True)
        g.predict_initial_states(allow_overlap=True)
        g.compute_fate_probabilities(use_petsc=True, n_jobs=-1)
        rna.obsm["fate_probabilities"] = pd.DataFrame(g.fate_probabilities.X,
                                                index = rna.obs_names,
                                                columns = g.fate_probabilities.names)
        mudata.obsm["rna_fates"] = rna.obsm["fate_probabilities"]
        mudata.obs["rna_pseudotime"] = rna.obs["pseudotime"].reindex(mudata.obs.index)
        compute_entropy(mudata = mudata,
                fate_probability_key = "rna_fates")

        mudata.obs["initial_states"] = g.initial_states.reindex(mudata.obs.index)
        mudata.obs["terminal_states"] = g.terminal_states.reindex(mudata.obs.index)
        mudata.obs["macrostates"] = g.macrostates.reindex(mudata.obs.index)
        failed = False
    except Exception as e:
        print(e)
        failed = True

    results = _apply_metrics(mudata = mudata, 
                            tree = tree, 
                            code = code, 
                            failed = failed, 
                            fixed_terminal = False,
                            terminal_clusters = TERM_DICT[tree],
                            true_probabilities = true_fates,
                            true_pseudotime_key = "rna:pseudotime",
                            pseudotime_key = "rna_pseudotime",
                            fate_key = "rna_fates",
                            cluster_key = "rna:pop",
                            seed = seed)

    _save_simulation_rna(
            tree = tree, 
            rd = diff_cif_fraction, 
            sigma= cif_sigma, 
            knn_rna = knn_rna, 
            fixed_terminal = False, 
            saving_folder = saving_simulation_path, 
            results=results)

    try:
        g = cellrank.estimators.GPCCA(kernel)
        g.compute_schur()
        g.set_initial_states(initial_cells)
        g.set_terminal_states(terminal_cells)
        g.compute_fate_probabilities(use_petsc=True, n_jobs=-1)
        rna.obsm["fate_probabilities"] = pd.DataFrame(g.fate_probabilities.X,
                                        index = rna.obs_names,
                                        columns = g.fate_probabilities.names)
        mudata.obsm["rna_fates"] = rna.obsm["fate_probabilities"]
        mudata.obs["rna_pseudotime"] = rna.obs["pseudotime"].reindex(mudata.obs.index)
        compute_entropy(mudata = mudata,
                fate_probability_key = "rna_fates")

        mudata.obs["initial_states"] = g.initial_states.reindex(mudata.obs.index)
        mudata.obs["terminal_states"] = g.terminal_states.reindex(mudata.obs.index)
        failed = False
    except Exception as e:
        print(e)
        failed = True

    results = _apply_metrics(mudata = mudata, 
                            tree = tree, 
                            code = code, 
                            failed = failed, 
                            fixed_terminal = True,
                            terminal_clusters = TERM_DICT[tree],
                            true_probabilities = true_fates,
                            true_pseudotime_key = "rna:pseudotime",
                            pseudotime_key = "rna_pseudotime",
                            fate_key = "rna_fates",
                            cluster_key = "rna:pop",
                            seed = seed)

    _save_simulation_rna(
            tree = tree, 
            rd = diff_cif_fraction, 
            sigma= cif_sigma, 
            knn_rna = knn_rna, 
            fixed_terminal = True, 
            saving_folder = saving_simulation_path, 
            results=results)
