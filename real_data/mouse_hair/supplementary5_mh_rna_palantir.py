########################################################################
# This code applies TI strategies on the SHARE-seq Mouse Hair data     #
# It presents scRNA-seq only based results                             # 
# Plots can be visualized in Supp. Figure 5                            #
########################################################################

import os
import atlas
import argparse
import pandas as pd
import numpy as np 
import muon as mu
import palantir
import cellrank
from muon import MuData
from atlas.tl.utils import compute_entropy, _assign_state_colors, _resolve_overlap, reset_state_colors
from real_data.utils import _compute_results, _get_plots, _states_to_columns, _save_metrics



if __name__ == "__main__":
    seed = 42
    np.random.seed(seed) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--knn_rna", type=int, default=15)
    parser.add_argument("--knn_act", type=int, default=15)
    parser.add_argument("--n_pcs_rna", type=int, default=20)
    parser.add_argument("--n_pcs_act", type=int, default=10)
    parser.add_argument("--wnn", default=None)
    args=parser.parse_args()

    pCode = f"{args.n_pcs_rna}:{args.n_pcs_act}_{args.knn_rna}:{args.knn_act}:{args.wnn}"
    code = f"rna_palantir_{pCode}"

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "mouse_hair")
    result_path = os.path.join(output_dir, "results.tsv")
    mudata = mu.read_h5mu(os.path.join( output_dir,
                            f"palantir_{pCode}.h5mu")
                    )

    # Select initial cell for Palantir computation to be the same aused in ATLAS
    early_cell = mudata.obs[~mudata.obs["initial_states"].isna()].index.values[0] 

    # PALANTIR ON RNA DATA
    rna = mudata["rna"]
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
                        fate_prob_key = "rna_fates")

    mudata.obsm["rna_fates"] = rna.obsm["rna_fates"].copy().reindex(mudata.obs.index)
    mudata.obs["rna_pseudotime"] = rna.obs["rna_pseudotime"].copy().reindex(mudata.obs.index)

    # ATLAS compute entropy overwrites the ATLAS entropy measures
    atlas_kl, atlas_she = mudata.obs["kl_divergence"].copy(), mudata.obs["shannon_entropy"].copy()
    compute_entropy(mudata = mudata,
            fate_probability_key = "rna_fates")

    # _assign_state_colors overwrites the ATLAS colors
    obs_is, obs_ms, obs_ts = mudata.obs["initial_states"].copy(), mudata.obs["macrostates"].copy(), mudata.obs["terminal_states"].copy() 
    atlas_colors = mudata.uns["atlas_state_palette"].copy()
    del mudata.uns["atlas_state_palette"]

    _terminal = {cell: [cell] for cell in rna.obsm["rna_fates"].columns}
    _initial = {early_cell : [early_cell]}
    _initial = _states_to_columns(_initial, mudata.obs_names)
    _terminal = _states_to_columns(_terminal, mudata.obs_names)
    mudata.obs["initial_states"] = _initial
    mudata.obs["terminal_states"] = _terminal
    mudata.obs["macrostates"] = _resolve_overlap(_initial, _terminal)
    _assign_state_colors(mudata)

    print("AVERAGE PSEUDOTIME PER CELL TYPE:")
    print(mudata.obs[["celltype", "rna_pseudotime"]].groupby("celltype").mean())

    print("AVERAGE PSEUDOTIME PER MACROSTATE:")
    print(mudata.obs[["macrostates", "rna_pseudotime"]].groupby("macrostates").mean())

    mudata.obs["rna_pseudotime"].to_csv(os.path.join(output_dir,f"rna_pseudotime_{pCode}.tsv"),
                                        sep = "\t",
                                        header = True, 
                                        index = True)

    # Compute metrics
    metrics = _compute_results(mudata = mudata,
                        code = code,
                        seed = seed, 
                        time_key = "rna_pseudotime",
                        fate_key = "rna_fates")
    _save_metrics(metrics, result_path)
    print("METRICS:")
    print(metrics)

    # Plots
    _get_plots(mudata = mudata,
            code = code,
            time_key = "rna_pseudotime",
            embedding_key = "umap",
            fate_key = "rna_fates",
            seed = seed)

    # Update the muon object and save withour redundancies 
    mudata.obs["rna_kl_divergence"] = mudata.obs["kl_divergence"]
    mudata.obs["rna_shannon_entropy"] = mudata.obs["shannon_entropy"]
    mudata.obs["kl_divergence"] = atlas_kl
    mudata.obs["shannon_entropy"] = atlas_she
    del mudata.obsm["rna_fates"]
    mudata.obs["rna_terminal_states"] = mudata.obs["terminal_states"]
    mudata.obs["rna_initial_states"] = mudata.obs["initial_states"]
    mudata.obs["rna_macrostates"] = mudata.obs["macrostates"]
    mudata.obs["macrostates"] = obs_ms
    mudata.obs["terminal_states"] = obs_ts
    mudata.obs["initial_states"] = obs_is
    mudata.uns["rna_state_palette"] = mudata.uns["atlas_state_palette"] 
    mudata.uns["atlas_state_palette"] = atlas_colors
    mudata.uns["rna_terminal_states_colors"] = mudata.uns["terminal_states_colors"]
    mudata.uns["rna_initial_states_colors"] = mudata.uns["initial_states_colors"]
    mudata.uns["rna_macrostates_colors"] = mudata.uns["macrostates_colors"]
    reset_state_colors(mudata)

    # cannot save if columns for .obsm slot are not strings
    rna.obsm["DM_EigenVectors"].columns = [str(c) for c in rna.obsm["DM_EigenVectors"].columns]
    rna.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in rna.obsm["DM_EigenVectors_multiscaled"].columns]
    mudata.write(os.path.join(output_dir,
                                f"{code}.h5mu"))
