import os
import json
import time
import fcntl
import atlas 
import argparse
import tracemalloc
import numpy as np
import pandas as pd
import muon as mu
from .utils import _compute_results, _get_plots
from atlas.tl import PalantirExtension, CellRankExtension


if __name__=="__main__":
    seed = 42
    threads = 3
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    #args 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--pcs_rna", type=int)
    parser.add_argument("--pcs_act", type=int)
    parser.add_argument("--knn_rna", type=int)
    parser.add_argument("--knn_activity", type=int)
    parser.add_argument("--wnn", type=int)
    args = parser.parse_args()
    knn_rna, knn_act, wnn = args.knn_rna, args.knn_activity, args.wnn
    n_pcs_rna, n_pcs_act= args.pcs_rna, args.pcs_act
    code = f"{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "embryonic_mouse_brain")
    results_path = os.path.join(output_dir, "results.csv")
    data_path = os.path.join(output_dir, "emb.h5mu")
    features_path = os.path.join(output_dir, "features.tsv")

    mudata = mu.read_h5mu(data_path)
    print(mudata)
    # select palantir initial cell
    initial = rng.choice(mudata[mudata.obs["celltype"]=="RG, Astro, OPC"].obs_names)
    features = pd.read_csv(features_path, sep = "\t", header=0, index_col=False)

    start_I_wall, start_I_cpu = time.perf_counter(), time.process_time()
    tracemalloc.start()
    mudata = atlas.pp.preprocessing(mudata = mudata,
                n_pcs_rna = n_pcs_rna, 
                n_pcs_act = n_pcs_act,
                knn_rna = knn_rna,
                knn_act = knn_act,
                n_neighbors = wnn,
                features = features)
    _, preprocessing_mem_peak = tracemalloc.get_traced_memory()
    preprocessing_mem_peak = preprocessing_mem_peak / (1024 * 1024)  # bytes -> MiB
    end_I_wall, end_I_cpu = time.perf_counter(), time.process_time()
    preprocessig_wall, preprocessing_cpu = end_I_wall - start_I_wall, end_I_cpu - start_I_cpu
    
    mu.tl.louvain(mudata)
    mu.pl.embedding(new_data, basis = "X_umap", color=["louvain", "celltype"], save ="{code}_cluster.png")

    print(mudata)
    exit()

    # TRAJECTORY INFERENCE USING PALANTIR
    try:
        start_P_wall, start_P_cpu = time.perf_counter(), time.process_time()
        tracemalloc.start()
        pex = PalantirExtension(mudata = mudata)
        pex.compute_kernel()
        pex.compute_diffusion_maps(seed = seed)
        pex.compute_multiscale_space()

        pex.run(early_cell = initial,
                    cluster_key = "celltype",
                    pseudotime_key = "pseudotime",
                    fate_prob_key = "palantir_probabilities",
                    terminal_states = None,
                    n_jobs = threads,
                    random_state = seed)

        _, palantir_mem_peak = tracemalloc.get_traced_memory()
        palantir_mem_peak = palantir_mem_peak / (1024 * 1024)  # bytes -> MiB
        end_P_wall, end_P_cpu = time.perf_counter(), time.process_time()
        palantir_wall, palantir_cpu = end_P_wall - start_P_wall, end_P_cpu - start_P_cpu

        palantir_metrics = _compute_results(mudata = mudata,
                                        code = code,
                                        seed = seed,
                                        time_key = "pseudotime",
                                        fate_key = "palantir_probabilities")
        with open(results_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            pd.DataFrame([palantir_metrics]).to_csv(f, index=False, header=f.tell() == 0)
            fcntl.flock(f, fcntl.LOCK_UN)
 
        _get_plots(mudata = mudata,
                code = code,
                time_key = "pseudotime",
                fate_key = "palantir_probabilities",
                seed = seed,
                ti_stategy = "palantir")

        # rename not to overwrite + adjust colnames for saving purposes 
        mudata.obs["palantir_SHE"] = mudata.obs["shannon_entropy"]
        mudata.obs["palantir_KL"] = mudata.obs["kl_divergence"]
        mudata.uns["palantir_terminal_states"] = mudata.uns["terminal_states"]
        mudata.uns["palantir_initial_states"] = mudata.uns["initial_states"]
        mudata.uns["palantir_fate_colors"] = mudata.uns["fate_state_colors"] 
        mudata.obsm["DM_EigenVectors"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors"].columns]
        mudata.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors_multiscaled"].columns]

    except Exception as e:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        palantir_wall, palantir_cpu, palantir_mem_peak = np.nan, np.nan, np.nan
        print(e)
        

    try:
        start_C_wall, start_C_cpu = time.perf_counter(), time.process_time()
        tracemalloc.start()
        cex = CellRankExtension(mudata=mudata)
        cex.compute_kernel(connectivity_key = "wnn_connectivities",
                        time_key = "pseudotime",
                        cluster_key = "celltype",
                        n_jobs = threads)
        cex.run(n_states = None, #automatically infer using minChi
            use_petsc = True,
            allow_overlap = True,
            n_jobs = threads)

        _, cellrank_mem_peak = tracemalloc.get_traced_memory()
        cellrank_mem_peak = cellrank_mem_peak / (1024 * 1024)  # bytes -> MiB
        end_C_wall, end_C_cpu = time.perf_counter(), time.process_time()
        cellrank_wall, cellrank_cpu = end_C_wall - start_C_wall, end_C_cpu - start_C_cpu

        mudata = cex.mudata
        pseudotimeK_metrics = _compute_results(mudata = mudata,
                                        code = code,
                                        seed = seed,
                                        time_key = "pseudotime",
                                        fate_key = "fate_probabilities")
        with open(results_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            pd.DataFrame([pseudotimeK_metrics]).to_csv(f, index=False, header=f.tell() == 0)
            fcntl.flock(f, fcntl.LOCK_UN)

        _get_plots(mudata = mudata,
                code = code,
                time_key = "pseudotime",
                fate_key = "fate_probabilities",
                seed = seed,
                ti_stategy = "pseudotimeK")


    except Exception as e:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        cellrank_wall, cellrank_cpu, cellrank_mem_peak = np.nan, np.nan, np.nan
        print(e)



    resources = {"code": code,
                "preprocessing_peak": preprocessing_mem_peak,
                "preprocessing_wall": preprocessing_wall,
                "preprocessing_cpu": preprocessing_cpu,
                "palantir_peak": palantir_mem_peak,
                "palantir_wall": palantir_wall,
                "palantir_cpu": palantir_cpu,
                "pseudotime_peak": cellrank_mem_peak,
                "pseudotime_wall": cellrank_wall,
                "pseudotime_cpu": cellrank_cpu,
    } 

    with open(resource_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pd.DataFrame([resources]).to_csv(f, index=False, header=f.tell() == 0)
        fcntl.flock(f, fcntl.LOCK_UN)

    
    mudata.write(os.path.join(output_dir, f"{code}.h5mu"))

