import os
import json
import atlas
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from muon import MuData
from anndata import AnnData
from atlas.tl import PalantirExtension
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation, _apply_metrics
from .supervised_metrics import terminal_state_score, js_distance, kendall_correlation



if __name__=="__main__":
    seed = 42
    working_directory = os.getcwd() # set path to repository 
    np.random.seed(seed)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tree", type=str)
    parser.add_argument("--rd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--knn_activity", type=int)
    parser.add_argument("--wnn", type=int, default=None)
    args = parser.parse_args()
    
    # DATA CONTRUCTION 
    tree = args.tree
    diff_cif_fraction, cif_sigma = args.rd, args.sigma
    knn_rna, knn_activity, wnn = args.knn_rna, args.knn_activity, args.wnn
    n_pcs_rna, n_pcs_activity = 20, 10
    
    data_path = os.path.join(working_directory, "data", "simulated_data", tree)
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "palantir")
    activity = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
    spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
    metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

    # create activity matrix
    activity = AnnData(X=csr_matrix(activity.values), 
            obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
            var = pd.DataFrame(data=None, index=activity.columns, columns=None))    
    sc.pp.normalize_total(activity)
    sc.pp.pca(activity, random_state=seed, use_highly_variable=False)
    # create rna matrix
    rna = AnnData(X=csr_matrix(spliced.values.T), 
            obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
            var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
    rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.pca(rna, random_state=seed, use_highly_variable=False)

    mudata = MuData({"rna":rna, "activity":activity})

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

    truth_dictionary = { "fate_probabilities" :true_fates, 
                        "true_states": {"initial" : early_cell,    
                                        "terminal": terminal_cells}
                        }

    # PREPROCESSING 
    atlas.pp.preprocessing(mudata = mudata,
                        n_pcs_rna=n_pcs_rna,
                        n_pcs_act=n_pcs_activity,
                        knn_rna=knn_rna,
                        knn_act=knn_activity,
                        n_neighbors=wnn,
                        random_state=seed)

    # INSTANTIATE CLASS
    pext = PalantirExtension(mudata = mudata)

    # RUN DIFFUSION COMPONENTS
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
                    random_state = seed)
        failed = False
    except Exception as e:
        print(e)
        failed = True

    ts_dict = mudata.uns.get("terminal_states", {})

    results = _apply_metrics(mudata = mudata, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
                    failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree],
                    ts_dict = ts_dict, true_probabilities = true_fates)
    _save_simulation(mudata = mudata, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = False, 
                    saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)

    # TRAJECTORY INFERENCE WITH TERMINAL STATES
    try: 
        pext.run(early_cell=early_cell[0],
                    cluster_key="rna:pop",
                    terminal_states=terminal_cells,
                    num_waypoints=num_waypoints)
        failed = False
    except Exception as e:
        print(e)
        failed = True
            
    ts_dict = mudata.uns.get("terminal_states", {})

    results = _apply_metrics(mudata = mudata, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
                    failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree],
                    ts_dict = ts_dict, true_probabilities = true_fates)
    _save_simulation(mudata = mudata, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
                        knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = True, 
                        saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)
