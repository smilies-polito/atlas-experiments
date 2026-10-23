import os
import warnings
import argparse
import palantir
import muon as mu
import numpy as np
import pandas as pd
from atlas.tl.utils import compute_entropy, _resolve_overlap, _disambiguate_names
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation_rna, _apply_metrics, _states_to_columns

if __name__=="__main__":
    seed = 42
    working_directory = os.getcwd() # set path to repository 
    np.random.seed(seed)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tree", type=str)
    parser.add_argument("--rd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    
    # DATA CONTRUCTION 
    tree = args.tree
    diff_cif_fraction, cif_sigma = args.rd, args.sigma
    knn_rna, knn_act, wnn = args.knn_rna, 30, 30
    n_pcs_rna, n_pcs_act = args.n_pcs_rna, 10
    muCode = f"{tree}_{diff_cif_fraction}:{cif_sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}"
    code = f"{tree}_{diff_cif_fraction}:{cif_sigma}_{n_pcs_rna}_{knn_rna}"
    
    data_path = os.path.join(working_directory, "output", "simulations", "mudata")
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "palantir_rna")
    mudata = mu.read_h5mu(os.path.join(data_path, f"{muCode}.h5mu"))
    mudata.pull_obs()

    # ground truth construction
    early_cell = initial_macrostate(mudata = mudata, 
            pseudotime_key = "rna:pseudotime", 
            n_cells=1)

    terminal_cells = []
    for branch in TERM_DICT[tree]:
        cell = terminal_macrostate(mudata = mudata,
                pseudotime_key = "rna:pseudotime", 
                cluster_key = "rna:pop",
                terminal_state = branch,
                n_cells=1)            
        terminal_cells.extend(cell)

    true_fates = truth_like_fates(
            pseudotime = mudata.obs["rna:pseudotime"],
            membership = mudata.obs["rna:pop"],
            tree = tree
    )

    # RUN WITH NO FIXED TERMINAL 
    rna = mudata["rna"].copy()
    try: 
        n_components, num_waypoints = 5, 250
        kernel = palantir.utils.compute_kernel(rna, knn=knn_rna)
        diffusion_maps = palantir.utils.diffusion_maps_from_kernel(kernel = kernel, n_components=n_components, seed = seed)
        rna.obsp["DM_Similarity"] = diffusion_maps["T"]
        rna.obsm["DM_EigenVectors"] = diffusion_maps["EigenVectors"].set_index(rna.obs.index)
        rna.uns["DM_EigenValues"] = diffusion_maps["EigenValues"].values
        eigenvectors = pd.DataFrame(rna.obsm["DM_EigenVectors"], index = rna.obs_names) if not isinstance(rna.obsm["DM_EigenVectors"], pd.DataFrame) else rna.obsm["DM_EigenVectors"]
        dm_dict = {"EigenValues": rna.uns["DM_EigenValues"], "EigenVectors": eigenvectors}
        result = palantir.utils.determine_multiscale_space(dm_res = dm_dict)
        rna.obsm["DM_EigenVectors_multiscaled"] = result
        palantir.core.run_palantir(data=rna,
                        early_cell = early_cell[0],
                        fate_prob_key = "rna_fates",
                        pseudo_time_key = "rna_pseudotime",
                        num_waypoints = num_waypoints,
                        n_jobs = args.n_jobs,
                        seed = seed)
        mudata.obsm["rna_fates"] = rna.obsm["rna_fates"].copy().reindex(mudata.obs.index)
        mudata.obs["rna_pseudotime"] = rna.obs["rna_pseudotime"].copy().reindex(mudata.obs.index)

        compute_entropy(mudata = mudata, 
                        fate_probability_key="rna_fates")
        
        proposed, terminal = {}, mudata.obsm["rna_fates"].columns
        for cell in early_cell:
            proposed[cell] = mudata.obs.loc[cell, "rna:pop"]
        for cell in terminal:
            proposed[cell] = mudata.obs.loc[cell, "rna:pop"]
        resolved = _disambiguate_names(proposed)

        _initial = {resolved[cell] : [cell] for cell in early_cell}
        _terminal = {resolved[cell] : [cell] for cell in terminal}
        _initial = _states_to_columns(_initial, mudata.obs_names)
        _terminal = _states_to_columns(_terminal, mudata.obs_names)
        mudata.obs["initial_states"] = _initial
        mudata.obs["terminal_states"] = _terminal
        mudata.obs["macrostates"] = _resolve_overlap(_initial, _terminal)
        failed = False
    except Exception as e:
        print(e)
        failed = True

    results = _apply_metrics(mudata = mudata, 
                tree = args.tree, 
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

    _save_simulation_rna(tree = tree, 
                        rd = diff_cif_fraction, 
                        sigma= cif_sigma, 
                        knn_rna = knn_rna, 
                        fixed_terminal = False, 
                        saving_folder = saving_simulation_path, 
                        results=results)

    # RUN WITH FIXED TERMINAL 
    rna = mudata["rna"].copy()
    try: 
        n_components, num_waypoints = 5, 250
        kernel = palantir.utils.compute_kernel(rna, knn=knn_rna)
        diffusion_maps = palantir.utils.diffusion_maps_from_kernel(kernel = kernel, n_components=n_components, seed = seed)
        rna.obsp["DM_Similarity"] = diffusion_maps["T"]
        rna.obsm["DM_EigenVectors"] = diffusion_maps["EigenVectors"].set_index(rna.obs.index)
        rna.uns["DM_EigenValues"] = diffusion_maps["EigenValues"].values
        eigenvectors = pd.DataFrame(rna.obsm["DM_EigenVectors"], index = rna.obs_names) if not isinstance(rna.obsm["DM_EigenVectors"], pd.DataFrame) else rna.obsm["DM_EigenVectors"]
        dm_dict = {"EigenValues": rna.uns["DM_EigenValues"], "EigenVectors": eigenvectors}
        result = palantir.utils.determine_multiscale_space(dm_res = dm_dict)
        rna.obsm["DM_EigenVectors_multiscaled"] = result
        palantir.core.run_palantir(data=rna,
                        early_cell = early_cell[0],
                        terminal_states = terminal_cells, 
                        fate_prob_key = "rna_fates",
                        pseudo_time_key = "rna_pseudotime",
                        num_waypoints = num_waypoints,
                        n_jobs = args.n_jobs,
                        seed = seed)
        mudata.obsm["rna_fates"] = rna.obsm["rna_fates"].copy().reindex(mudata.obs.index)
        mudata.obs["rna_pseudotime"] = rna.obs["rna_pseudotime"].copy().reindex(mudata.obs.index)

        compute_entropy(mudata = mudata, 
                        fate_probability_key="rna_fates")
        
        proposed, terminal = {}, mudata.obsm["rna_fates"].columns
        for cell in early_cell:
            proposed[cell] = mudata.obs.loc[cell, "rna:pop"]
        for cell in terminal:
            proposed[cell] = mudata.obs.loc[cell, "rna:pop"]
        resolved = _disambiguate_names(proposed)

        _initial = {resolved[cell] : [cell] for cell in early_cell}
        _terminal = {resolved[cell] : [cell] for cell in terminal}
        _initial = _states_to_columns(_initial, mudata.obs_names)
        _terminal = _states_to_columns(_terminal, mudata.obs_names)
        mudata.obs["initial_states"] = _initial
        mudata.obs["terminal_states"] = _terminal
        mudata.obs["macrostates"] = _resolve_overlap(_initial, _terminal)
        mudata.obsm["rna_fates"].columns = [resolved[cell] for cell in terminal]
        failed = False
    except Exception as e:
        print(e)
        failed = True

    results = _apply_metrics(mudata = mudata, 
                tree = args.tree, 
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

    _save_simulation_rna(tree = tree, 
                        rd = diff_cif_fraction, 
                        sigma= cif_sigma, 
                        knn_rna = knn_rna, 
                        fixed_terminal = True, 
                        saving_folder = saving_simulation_path, 
                        results=results)
