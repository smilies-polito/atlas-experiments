import os
import argparse
import muon as mu
import numpy as np
from atlas.tl import CellRankExtension
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
    parser.add_argument("--wnn", type=int, default = None)
    args = parser.parse_args()
    
    # DATA CONSTRUCTION
    tree = args.tree
    diff_cif_fraction, cif_sigma = args.rd, args.sigma
    knn_rna, knn_act, wnn = args.knn_rna, args.knn_act, args.wnn
    n_pcs_rna, n_pcs_act= 20, 10
    pCode = f"{tree}_{diff_cif_fraction}:{cif_sigma}_{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{args.wnn}"
    
    data_path = os.path.join(working_directory, "output", "simulations", "mudata")
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "pseudotime_kernel")
    mudata = mu.read_h5mu(os.path.join(data_path, f"{pCode}.h5mu"))
    mudata.pull_obs()

    early_cell = initial_macrostate(mudata = mudata, 
                                    pseudotime_key = "rna:pseudotime", 
                                    n_cells=30)
    cluster_subset = mudata.obs["rna:pop"].loc[early_cell].value_counts().idxmax()
    early_cell = {cluster_subset : early_cell}

    terminal_cells = {}
    for branch in TERM_DICT[tree]:
        cell = terminal_macrostate(mudata = mudata,
                                    pseudotime_key = "rna:pseudotime", 
                                    cluster_key = "rna:pop",
                                    terminal_state = branch,
                                    n_cells=30)
        terminal_cells[branch] = cell

    true_fates = truth_like_fates(pseudotime = mudata.obs["rna:pseudotime"],
                                    membership = mudata.obs["rna:pop"],
                                    tree = tree)


    cext = CellRankExtension(mudata=mudata.copy())
    try:
        cext.compute_kernel(key = "wnn",
                            time_key = "rna:pseudotime",
                            cluster_key = "rna:pop",
                            threshold_scheme = "hard")
        failed = False
    except Exception as e:
        print(e)
        failed = True

    # TI WITH NO FIXED TERMINAL
    try:
        cext.run(
            n_states = None, 
            n_jobs = -1, 
            allow_overlap = True)
        failed = False
    except Exception as e: 
        print(e)
        failed = True

    data = cext.mudata 
    results = _apply_metrics(mudata = data, 
                               tree = tree, 
                               code = pCode,
                               failed = failed,
                               fixed_terminal = False,
                               terminal_clusters = TERM_DICT[tree],
                               true_probabilities = true_fates,
                               true_pseudotime_key = "rna:pseudotime",
                               pseudotime_key = "rna:pseudotime",
                               cluster_key = "rna:pop",
                               seed = seed)

    _save_simulation(mudata = data, 
                    tree = tree, 
                    rd = diff_cif_fraction, 
                    sigma= cif_sigma, 
                    knn_rna = knn_rna, 
                    knn_act= knn_act, 
                    wnn= wnn, 
                    fixed_terminal = False, 
                    saving_folder = saving_simulation_path, 
                    results=results)


    # RUN WITH FIXED TERMINAL
    cext = CellRankExtension(mudata=mudata.copy())
    try:
        cext.compute_kernel(key = "wnn",
                            time_key = "rna:pseudotime",
                            cluster_key = "rna:pop",
                            threshold_scheme = "hard")
        failed = False
    except Exception as e:
        print(e)
        failed = True

    try: 
        cext.run(
            n_jobs = -1,
            initial_states = early_cell, 
            terminal_states = terminal_cells)
        failed = False
    except Exception as e:
        print(e)
        failed = True

    data = cext.mudata 
    results = _apply_metrics(mudata = data, 
                               tree = tree, 
                               code = pCode,
                               failed = failed,
                               fixed_terminal = True,
                               terminal_clusters = TERM_DICT[tree],
                               true_probabilities = true_fates,
                               true_pseudotime_key = "rna:pseudotime",
                               pseudotime_key = "rna:pseudotime",
                               cluster_key = "rna:pop",
                               seed = seed)

    _save_simulation(mudata = data, 
                    tree = tree, 
                    rd = diff_cif_fraction, 
                    sigma= cif_sigma, 
                    knn_rna = knn_rna, 
                    knn_act= knn_act, 
                    wnn= wnn, 
                    fixed_terminal = True,
                    saving_folder = saving_simulation_path, 
                    results=results)

