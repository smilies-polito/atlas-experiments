import os
import json
import atlas
import fcntl
import time
import tracemalloc
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from muon import MuData
from anndata import AnnData
from atlas.tl import PalantirExtension
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT
from .supervised_metrics import terminal_state_score, js_distance, kendall_correlation 

def _save_simulation(
                    mudata: MuData,
                    tree: str,
                    rd: float,     
                    sigma: float,
                    knn_rna: int, 
                    knn_activity: int,
                    wnn: int,
                    fixed_terminal: bool, 
                    results: dict,
                    ground_truth: dict,
                    saving_folder: str,
                    resources: dict):
    '''
        Save simulation results.
    '''
    code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}:{knn_activity}:{wnn}"
    del mudata.mod["activity"].obsm
    del mudata.mod["activity"].obsp
    del mudata.mod["activity"].uns
    del mudata.mod["activity"].varm
    mudata.mod["activity"].X = None
    del mudata.mod["rna"].obsm
    del mudata.mod["rna"].obsp
    del mudata.mod["rna"].uns
    del mudata.mod["rna"].varm
    mudata.mod["activity"].X = None

    mudata.obsm["true_fates"] = ground_truth["fate_probabilities"]
    mudata.uns["true_states"] = ground_truth["true_states"]
    mudata.obsm["DM_EigenVectors"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors"].columns]
    mudata.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors_multiscaled"].columns]
    mudata.uns["simulation_results"] = results
    mudata.write(os.path.join(saving_folder, f"{code}.h5mu"))

    results_path = os.path.join(saving_folder, "results.csv")
    resources_path = os.path.join(saving_folder, "resources.csv")
    resources["code"] = code

    with open(results_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pd.DataFrame([results]).to_csv(f, index=False, header=f.tell() == 0)
        fcntl.flock(f, fcntl.LOCK_UN)

    with open(resources_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pd.DataFrame([resources]).to_csv(f, index=False, header=f.tell() == 0)
        fcntl.flock(f, fcntl.LOCK_UN)
    
    

def _apply_metrics_and_visualize(mudata: MuData,
                                tree: str,
                                rd: float, 
                                sigma: float,
                                knn_rna: int, 
                                knn_activity: int,
                                wnn: int,
                                ts_dict: dict,
                                terminal_clusters: list,
                                true_probabilities: pd.DataFrame,
                                failed: bool = False,
                                fixed_terminal: bool = False,
                                seed: int = 42
                                ):
    '''
        Compute metrics.
    '''
    code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}:{knn_activity}:{wnn}"
    results = {
                "code" : code,
                "failed": failed,
                "fixed_terminal" : fixed_terminal,
                "spearman_stat_pseudotime": np.nan,
                "spearman_pval_pseudotime": np.nan,
                "kendall_stat_pseudotime": np.nan,
                "kendall_pval_pseudotime": np.nan,
                "fate_index_pval": np.nan,
                "fate_index_stat": np.nan,
                "n_terminal_states" : np.nan,
                "pearson_pval_KLD": np.nan,
                "pearson_stat_KLD": np.nan,
                "pearson_pval_SHE": np.nan,
                "pearson_stat_SHE": np.nan,
                "spearman_pval_KLD": np.nan,
                "spearman_stat_KLD": np.nan,
                "spearman_pval_SHE": np.nan,
                "spearman_stat_SHE": np.nan,
                "temporal_state_score": np.nan,
                "terminal_enrichment": np.nan,
                "terminal_silhouette_pse": np.nan,
                "terminal_silhouette_soft": np.nan,
                "tsr": np.nan,
                "ttc": np.nan,
                "ttp": np.nan,
                "tts": np.nan,
                "jsd_totipotent": np.nan,
                "jsd_multipotent": np.nan,
                "jsd_committed": np.nan,
            }

    if failed:
        return results
                    
    results["n_terminal_states"] = mudata.obsm["fate_probabilities"].shape[1]

    # SUPERVISED - correlation true and inferred pseudotime
    stat, pval, _ = atlas.tl.spearman_correlation(mudata = mudata, 
                                                    key1 = "pseudotime",
                                                    key2 = "rna:pseudotime",
                                                    seed = seed
                                                 )
    results["spearman_stat_pseudotime"] = stat
    results["spearman_pval_pseudotime"] = pval
    stat, pval =  kendall_correlation(mudata = mudata,
                                        pred_time = "pseudotime",
                                        true_time = "rna:pseudotime")
    results["kendall_stat_pseudotime"] = stat
    results["kendall_pval_pseudotime"] = pval

    # no fate probabilities    
    if results["n_terminal_states"] <= 0:
        return results    

    #SUPERVISED - JSD
    if fixed_terminal:
        jsd = js_distance(mudata = mudata,
                            truth = true_probabilities,
                            fate_key = "fate_probabilities"
                        )
                            
        if not jsd.index.equals(mudata.obs["rna:pop"].index):
            jsd = jsd.loc[mudata.obs["rna:pop"].index]
        jsdf = pd.DataFrame({"jsd": jsd, "cluster": mudata.obs["rna:pop"]})
        jsdf["potency"] = jsdf["cluster"].map(POTENCY_DICT[tree])
        mean_jsd = jsdf.groupby("potency")["jsd"].mean()
        for cat in ["multipotent", "totipotent", "committed"]:
            results[f"jsd_{cat}"] = mean_jsd.get(cat, np.nan)

    #SUPERVISED - TERMINAL STATE SCORE
    tts, ttp, tsr, ttc, overall = terminal_state_score(mudata = mudata,
                                                        time_key = "pseudotime",
                                                        cluster_key = "rna:pop",
                                                        terminal_clusters = terminal_clusters)
    results["tts"] = tts
    results["ttp"] = ttp
    results["tsr"] = tsr
    results["ttc"] = ttc
    results["temporal_state_score"] = overall

    #UNSUPERVISED
    stat, pval, _ = atlas.tl.spearman_correlation(
            mudata = mudata,
            key1 = "pseudotime",
            key2 = "kl_divergence",
            seed = seed)
    results["spearman_stat_KLD"] = stat
    results["spearman_pval_KLD"] = pval
    stat, pval, _ = atlas.tl.spearman_correlation(
            mudata = mudata,
            key1 = "pseudotime",
            key2 = "shannon_entropy",
            seed = seed)
    results["spearman_stat_SHE"] = stat
    results["spearman_pval_SHE"] = pval
    stat, pval, _ = atlas.tl.pearson_correlation(
            mudata = mudata,
            key1 = "pseudotime",
            key2 = "kl_divergence",
            seed = seed)
    results["pearson_stat_KLD"] = stat
    results["pearson_pval_KLD"] = pval
    stat, pval, _ = atlas.tl.pearson_correlation(
            mudata = mudata,
            key1 = "pseudotime",
            key2 = "shannon_entropy",
            seed = seed)
    results["pearson_stat_SHE"] = stat
    results["pearson_pval_SHE"] = pval

    stat, pval, _, __ = atlas.tl.fate_concentration_index(
            mudata = mudata,
            fate_key = "fate_probabilities",
            time_key = "pseudotime",
            seed = seed)
    results["fate_index_stat"] = stat
    results["fate_index_pval"] = pval

    results["terminal_silhouette_soft"] = atlas.tl.terminal_state_silhouette(
            mudata = mudata,
            fate_key = "fate_probabilities",
            soft_assignment = True)
    results["terminal_silhouette_pse"] = atlas.tl.terminal_state_silhouette(
            mudata = mudata,
            fate_key = "fate_probabilities",
            time_key = "pseudotime",
            soft_assignment = False)

    results["terminal_enrichment"] = atlas.tl.terminal_pseudotime_enrichment(
                                            mudata = mudata,
                                            time_key = "pseudotime",
                                            rank = True)

    # VISUALIZATION
    atlas.pl.plot_embedding(mudata = mudata,
                        embedding_key = "X_umap",
                        observation = "pseudotime",
                        save = f"_PSTIME{code}.png",
                        show= False)
    atlas.pl.plot_embedding(mudata = mudata,
                        embedding_key = "X_umap",
                        observation = "kl_divergence",
                        save = f"_KLDIV_{code}.png",
                        show= False)
    atlas.pl.plot_embedding(mudata = mudata,
                        embedding_key = "X_umap",
                        observation = "shannon_entropy",
                        save = f"_SHENTR_{code}.png",
                        show= False)
    atlas.pl.plot_fate_probabilities(mudata = mudata,
                                    fate_probability_key = "fate_probabilities",
                                    embedding_key= "X_umap",
                                    states = None,
                                    save =  f"_fates_{code}.png",
                                    show= False)
    try:
        atlas.pl.plot_tree(mudata = mudata,
                        embedding_key = "umap",
                        fate_probability_key = "fate_probabilities",
                        save = f"_{code}.png",
                        color = "rna:pop", 
                        color_milestones = False,
                        show = False)
    except (IndexError, KeyError, ValueError) as e:
        print(f"WARNING: plot_tree failed for {code}: {e}")                    
    return results
    


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
    parser.add_argument("--wnn", type=int)
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
                                    tree = args.tree)

    truth_dictionary = { "fate_probabilities" :true_fates, 
                        "true_states": {"initial" : early_cell,    
                                        "terminal": terminal_cells}
                        }

    # PREPROCESSING 
    start_p_wall, start_p_cpu = time.perf_counter(), time.process_time()
    tracemalloc.start()
    atlas.pp.preprocessing(mudata = mudata,
                        n_pcs_rna=n_pcs_rna,
                        n_pcs_act=n_pcs_activity,
                        knn_rna=knn_rna,
                        knn_act=knn_activity,
                        n_neighbors=wnn,
                        random_state=seed)
    _, preprocessing_mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    preprocessing_mem_peak = preprocessing_mem_peak / (1024 * 1024)  
    end_p_wall, end_p_cpu = time.perf_counter(), time.process_time()
    preprocessing_wall, preprocessing_cpu = end_p_wall - start_p_wall, end_p_cpu - start_p_cpu

    # INSTANTIATE CLASS
    start_i_wall, start_i_cpu = time.perf_counter(), time.process_time()
    tracemalloc.start()
    pext = PalantirExtension(mudata = mudata)
    _, init_mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    init_mem_peak = init_mem_peak / (1024 * 1024)  
    end_i_wall, end_i_cpu = time.perf_counter(), time.process_time()
    init_wall, init_cpu = end_i_wall - start_i_wall, end_i_cpu - start_i_cpu

    # RUN DIFFUSION COMPONENTS
    try: 
        n_components, num_waypoints = 5, 250
        start_k_wall, start_k_cpu = time.perf_counter(), time.process_time()
        tracemalloc.start()
        pext.compute_kernel()
        pext.compute_diffusion_map(n_components = n_components, seed = seed)
        pext.compute_multiscale_space()
        _, kernel_mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        kernel_mem_peak = kernel_mem_peak / (1024 * 1024)  # bytes -> MiB
        end_k_wall, end_k_cpu = time.perf_counter(), time.process_time()
        kernel_wall, kernel_cpu = end_k_wall - start_k_wall, end_k_cpu - start_k_cpu
        failed = False
    except Exception as e:
        print(e)
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        failed = True
        kernel_wall, kernel_cpu, kernel_mem_peak = np.nan, np.nan, np.nan

    # TRAJECTORY INFERENCE WITHOUT TERMINAL STATES
    try:
        start_r_wall, start_r_cpu = time.perf_counter(), time.process_time()
        tracemalloc.start()
        pext.run(early_cell=early_cell[0],
                    cluster_key="rna:pop",
                    terminal_states=None,
                    num_waypoints=num_waypoints,
                    random_state = seed)
        _, run_mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        run_mem_peak = run_mem_peak / (1024 * 1024)  # bytes -> MiB
        end_r_wall, end_r_cpu = time.perf_counter(), time.process_time()
        run_wall, run_cpu = end_r_wall - start_r_wall, end_r_cpu - start_r_cpu
        failed = False
    except Exception as e:
        print(e)
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        failed = True
        run_wall, run_cpu, run_mem_peak = np.nan, np.nan, np.nan

    ts_dict = pext.mudata.uns.get("terminal_states", {})
    resources = {"init_wall_time": init_wall,
                    "init_cpu_time": init_cpu,
                    "preprocessing_wall_time": preprocessing_wall,
                    "preprocessinf_cpu_time": preprocessing_cpu,
                    "run_wall_time": kernel_wall + run_wall,
                    "run_cpu_time": kernel_cpu + run_cpu,
                    "init_mem_peak": init_mem_peak,
                    "run_mem_peak": np.maximum(kernel_mem_peak, run_mem_peak),
                    "preprocessing_mem_peak": preprocessing_mem_peak
                }

    results = _apply_metrics_and_visualize(mudata = mudata, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
                    failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree],
                    ts_dict = ts_dict, true_probabilities = true_fates)
    _save_simulation(mudata = mudata, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = False, 
                    saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary, resources=resources)

    # TRAJECTORY INFERENCE WITH TERMINAL STATES
    try: 
        start_r_wall, start_r_cpu = time.perf_counter(), time.process_time()
        tracemalloc.start()
        pext.run(early_cell=early_cell[0],
                    cluster_key="rna:pop",
                    terminal_states=terminal_cells,
                    num_waypoints=num_waypoints)
        _, run_mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        run_mem_peak = run_mem_peak / (1024 * 1024)  # bytes -> MiB
        end_r_wall, end_r_cpu = time.perf_counter(), time.process_time()
        run_wall, run_cpu = end_r_wall - start_r_wall, end_r_cpu - start_r_cpu
        failed = False
    except Exception as e:
        print(e)
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        failed = True
        run_wall, run_cpu, run_mem_peak = np.nan, np.nan, np.nan
            
    ts_dict = pext.mudata.uns.get("terminal_states", {})
    resources = {"init_wall_time": init_wall,
                    "init_cpu_time": init_cpu,
                    "preprocessing_wall_time": preprocessing_wall,
                    "preprocessinf_cpu_time": preprocessing_cpu,
                    "run_wall_time": run_wall + kernel_wall,
                    "run_cpu_time": run_cpu + kernel_cpu,
                    "init_mem_peak": init_mem_peak,
                    "run_mem_peak": np.maximum(run_mem_peak, kernel_mem_peak),
                    "preprocessing_mem_peak": preprocessing_mem_peak
                }
    results = _apply_metrics_and_visualize(mudata = mudata, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
                    knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
                    failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree],
                    ts_dict = ts_dict, true_probabilities = true_fates)
    _save_simulation(mudata = mudata, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
                        knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = True, 
                        saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary, resources=resources)
