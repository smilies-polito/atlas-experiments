import os
import json
import atlas
import argparse
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from muon import MuData
from anndata import AnnData
from atlas.tl import CellRankExtension
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation, _apply_metrics
from .supervised_metrics import terminal_state_score, js_distance, kendall_correlation


if __name__=="__main__":
    seed = 42
    threads = 3
    working_directory = os.getcwd() # set path to repository 
    np.random.seed(seed)

    parser = argparse.ArgumentParser() 
    parser.add_argument("--tree", type=str)
    parser.add_argument("--rd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--knn_activity", type=int)
    parser.add_argument("--wnn", type=int, default = None)
    args = parser.parse_args()
    
    # DATA CONSTRUCTION
    tree = args.tree
    diff_cif_fraction, cif_sigma = args.rd, args.sigma
    knn_rna, knn_activity, wnn = args.knn_rna, args.knn_activity, args.wnn
    n_pcs_rna, n_pcs_activity = 20, 10
    
    data_path = os.path.join(working_directory, "data", "simulated_data", tree)
    saving_simulation_path = os.path.join(working_directory, "output", "simulations", "pseudotime_kernel")
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

    truth_dictionary = { "fate_probabilities" :true_fates, 
                        "true_states": {"initial" : early_cell,    
                                        "terminal": terminal_cells}
                        }


    # PREPROCESSING
    atlas.pp.preprocessing(mudata = mudata,
                        n_pcs_rna = n_pcs_rna,
                        n_pcs_act = n_pcs_activity,
                        knn_rna = knn_rna,
                        knn_act = knn_activity,
                        n_neighbors = wnn,
                        random_state = seed) 

    # INITIALIZATION
    cext = CellRankExtension(mudata=mudata)

    # COMPUTE KERNEL
    try:
        cext.compute_kernel(connectivity_key = "wnn_connectivities",
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
            n_jobs = threads, 
            allow_overlap = True)
        failed = False
    except Exception as e: 
        failed = True

    ts_dict = mudata.uns.get("terminal_states", {})

    results = _apply_metrics(mudata = mudata, tree = tree, rd= diff_cif_fraction, sigma = cif_sigma,
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn, pseudotime_key = "rna:pseudotime",
                    failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree],
                    ts_dict = ts_dict, true_probabilities = true_fates, ground_truth_pseudotime=False)
    _save_simulation(mudata = mudata, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = False, 
                    saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)


    # RUN WITH FIXED TERMINAL
    try: 
        cext.run(
            n_jobs = threads,
            initial_states = early_cell, 
            terminal_states = terminal_cells)
        failed = False
    except:
        failed = True

    ts_dict = mudata.uns.get("terminal_states", {})

    results = _apply_metrics(mudata = mudata, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
            knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn, pseudotime_key = "rna:pseudotime",
                    failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree],
                    ts_dict = ts_dict, true_probabilities = true_fates, ground_truth_pseudotime=False)
    _save_simulation(mudata = mudata, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = True, 
                    saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)
