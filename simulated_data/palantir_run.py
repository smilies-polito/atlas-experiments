import os
import muon as mu
import json
import argparse
import numpy as np
from atlas.tl import PalantirExtension
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation, _apply_metrics


if __name__=="__main__":
    seed = 42
    working_directory = os.getcwd() # set path to repository 
    np.random.seed(seed)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tree", type=str)
    parser.add_argument("--rd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--knn_act", type=int)
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--wnn", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()
    
    # DATA CONTRUCTION 
    tree = args.tree
    diff_cif_fraction, cif_sigma = args.rd, args.sigma
    knn_rna, knn_act, wnn = args.knn_rna, args.knn_act, args.wnn
    n_pcs_rna, n_pcs_act = args.n_pcs_rna, args.n_pcs_act
    pCode = f"{tree}_{diff_cif_fraction}:{cif_sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{args.wnn}"
    
    data_path = os.path.join(working_directory, "output", "simulations", "mudata")
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "palantir")
    mudata = mu.read_h5mu(os.path.join(data_path, f"{pCode}.h5mu"))
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

    true_fates = truth_like_fates(pseudotime = mudata.obs["rna:pseudotime"],
                                    membership = mudata.obs["rna:pop"],
                                    tree = tree)

    pext = PalantirExtension(mudata = mudata.copy())
    try: 
        n_components, num_waypoints = 5, 250
        pext.compute_kernel()
        pext.compute_diffusion_map(n_components = n_components, seed = seed)
        pext.compute_multiscale_space()
        failed = False
    except Exception as e:
        print(e)
        failed = True

    # TRAJECTORY INFERENCE WITHOUT TERMINAL STATES
    try:
        pext.run(early_cell=early_cell[0],
                    cluster_key="rna:pop",
                    terminal_states=None,
                    num_waypoints=num_waypoints,
                    n_jobs=args.n_jobs,
                    random_state = seed)
        failed = False
    except Exception as e:
        print(e)
        failed = True

    data = pext.mudata 
    results = _apply_metrics(mudata = data,
                            tree = tree,
                            code = pCode,
                            failed = failed,
                            fixed_terminal = False, 
                            terminal_clusters = TERM_DICT[tree], 
                            true_probabilities = true_fates,
                            true_pseudotime_key = "rna:pseudotime",
                            cluster_key = "rna:pop",
                            seed = seed)


    _save_simulation(mudata = data, 
                    tree = tree, 
                    rd = diff_cif_fraction, 
                    sigma= cif_sigma, 
                    knn_rna = knn_rna,
                    knn_act = knn_act, 
                    wnn= wnn, 
                    fixed_terminal = False, 
                    saving_folder = saving_simulation_path, 
                    results=results)

    # TRAJECTORY INFERENCE WITH TERMINAL STATES
    pext = PalantirExtension(mudata = mudata.copy())
    try: 
        pext.run(early_cell=early_cell[0],
                    cluster_key="rna:pop",
                    terminal_states=terminal_cells,
                    num_waypoints=num_waypoints,
                    n_jobs=args.n_jobs)
        failed = False
    except Exception as e:
        print(e)
        failed = True
            
    data = pext.mudata 
    results = _apply_metrics(mudata = data,
                            tree = tree,
                            code = pCode,
                            failed = failed,
                            fixed_terminal = True, 
                            terminal_clusters = TERM_DICT[tree], 
                            true_probabilities = true_fates,
                            true_pseudotime_key = "rna:pseudotime",
                            cluster_key = "rna:pop",
                            seed = seed)


    _save_simulation(mudata = data, 
                    tree = tree, 
                    rd = diff_cif_fraction, 
                    sigma= cif_sigma, 
                    knn_rna = knn_rna,
                    knn_act = knn_act, 
                    wnn= wnn, 
                    fixed_terminal = True, 
                    saving_folder = saving_simulation_path,
                    results=results)
