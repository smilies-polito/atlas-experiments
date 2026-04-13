import os
import json
import time
import atlas 
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

    knn_rna, knn_activity, wnn = None
    n_pcs_rna, n_pcs_act= 20, 10
    code = f"{n_pcs_rna}:{n_pcs_act}_{knn_rna}:{knn_act}:{wnn}_hard"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "mouse_skin")
    results_path = os.path.join(output_dir, "results.csv")
    resource_path = os.path.join(output_dir, "resources.csv")
    data_path = os.path.join(output_dir, "skin.h5mu")

    mudata = mu.read_h5mu(data_path)
    # select palantir initial cell
    initial = rng.choice(mudata[mudata.obs["celltype"]=="TAC-1"].obs_names)

    new_data = atlas.pp.preprocessing(mudata = mudata,
                n_pcs_rna = n_pcs_rna, 
                n_pcs_act = n_pcs_act,
                knn_rna = knn_rna,
                knn_act = knn_act,
                n_neighbors = wnn,
                features = None)
    
    mu.tl.louvain(new_data)
    mu.pl.embedding(new_data, basis = "X_umap", color=["louvain", "celltype"], save =f"{code}_cluster.png")


    pex = PalantirExtension(mudata = new_data)
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

    palantir_metrics = _compute_results(mudata = new_data,
                                    code = code,
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "palantir_probabilities")

    _get_plots(mudata = new_data,
            code = code,
            time_key = "pseudotime",
            fate_key = "palantir_probabilities",
            seed = seed,
            ti_strategy = "palantir")

    # rename not to overwrite + adjust colnames for saving purposes 
    new_data.obs["palantir_SHE"] = new_data.obs["shannon_entropy"]
    new_data.obs["palantir_KL"] = new_data.obs["kl_divergence"]
    new_data.uns["palantir_terminal_states"] = new_data.uns["terminal_states"]
    new_data.uns["palantir_initial_states"] = new_data.uns["initial_states"]
    new_data.uns["palantir_fate_colors"] = new_data.uns["fate_state_colors"] 
    new_data.obsm["DM_EigenVectors"].columns = [str(c) for c in new_data.obsm["DM_EigenVectors"].columns]
    new_data.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in new_data.obsm["DM_EigenVectors_multiscaled"].columns]


    cex = CellRankExtension(mudata=new_data)
    cex.compute_kernel(connectivity_key = "wnn_connectivities",
                    time_key = "pseudotime",
                    cluster_key = "celltype",
                    n_jobs = -1)
    cex.run(n_states = None, 
        use_petsc = True,
        allow_overlap = True,
        n_jobs = -1)

    pseudotimeK_metrics = _compute_results(mudata = new_data,
                                    code = code,
                                    seed = seed,
                                    time_key = "pseudotime",
                                    fate_key = "fate_probabilities")

    _get_plots(mudata = new_data,
            code = code,
            time_key = "pseudotime",
            fate_key = "fate_probabilities",
            seed = seed,
            ti_strategy = "pseudotimeK")

    path = os.path.join(output_dir, f"{code}.h5mu")
    new_data.write(os.path.join(output_dir, f"{code}.h5mu"))

    print(palantir_metrics)
    print(pseudotimeK_metrics)
