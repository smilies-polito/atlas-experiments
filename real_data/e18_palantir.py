import os
import atlas 
import numpy as np
import muon as mu
from atlas.tl import PalantirExtension 

if __name__=="__main__":
    seed = 42
    rng = np.random.default_rng(seed)

    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, "output", "embryonic_mouse_brain")
    data_path = os.path.join(output_dir, "emb.h5mu")

    mudata = mu.read_h5mu(data_path)

    pex = PalantirExtension(mudata = mudata)
    pex.compute_kernel()
    pex.compute_diffusion_map(seed = seed)
    pex.compute_multiscale_space()

    #select initial cell
    initial = rng.choice(mudata[mudata.obs["celltype"]=="RG, Astro, OPC"].obs_names)
    pex.run(early_cell = initial,
            cluster_key = "celltype",
            terminal_states = None,
            n_jobs = -1,
            random_state = seed)

    mudata = pex.mudata

    results = {}
    stat, pval, ci = atlas.tl.pearson_correlation(mudata = mudata,
                                                key1 = "pseudotime",
                                                key2 = "kl_divergence",
                                                seed = seed)
    results["pcorr_stat_KL"] = stat
    results["pcorr_pval_KL"] = pval
    results["ci_KL"] = ci

    stat, pval, ci = atlas.tl.pearson_correlation(mudata = mudata,
                                                key1 = "pseudotime",
                                                key2 = "shannon_entropy",
                                                seed = seed)
    results["pcorr_stat_SHE"] = stat
    results["pcorr_pval_SHE"] = pval
    results["ci_SHE"] = ci

    stat, pval, ci = atlas.tl.spearman_correlation(mudata = mudata,
                                                key1 = "pseudotime",
                                                key2 = "kl_divergence",
                                                seed = seed)
    results["scorr_stat_KL"] = stat
    results["scorr_pval_KL"] = pval
    results["scorr_ci_KL"] = ci

    stat, pval, ci = atlas.tl.spearman_correlation(mudata = mudata,
                                                key1 = "pseudotime",
                                                key2 = "shannon_entropy",
                                                seed = seed)
    results["scorr_stat_SHE"] = stat
    results["scorr_pval_SHE"] = pval
    results["scorr_ci_SHE"] = ci

    stat, pval, ci, fci = atlas.tl.fate_concentration_index(mudata=mudata,
                                                        time_key = "pseudotime",
                                                        fate_key = "fate_probabilities",
                                                        seed = seed)
    results["fci_stat"] = stat
    results["fci_pval"] = pval
    results["fci_ci"] = ci

    results["pseudotime_enrichment"] = atlas.tl.terminal_pseudotime_enrichment(
                                                            mudata = mudata,
                                                            time_key = "pseudotime",
                                                            rank = True)

    results["silhouette_soft"] = atlas.tl.terminal_state_silhouette(mudata = mudata,
                                                                    fate_key = "fate_probabilities",
                                                                    soft_assignment= True)
    results["sihouette_hard"] = atlas.tl.terminal_state_silhouette(mudata = mudata,
                                                                    fate_key = "fate_probabilities",
                                                                    soft_assignment = False,
                                                                    time_key = "pseudotime")
    atlas.pl.plot_embedding(mudata = mudata,
                            embedding_key = "X_umap",
                            observation = "pseudotime",
                            save =  "_pseudotime.png")


    atlas.pl.plot_embedding(mudata = mudata,
                            embedding_key = "X_umap",
                            observation = "shannon_entropy",
                            save = "_shannon_entropy.png")


    atlas.pl.plot_embedding(mudata = mudata,
                            embedding_key = "X_umap",
                            observation = "kl_divergence",
                            save = "_kl_divergence.png")
                        
    atlas.pl.plot_fate_probabilities(mudata = mudata,
                                        embedding_key = "X_umap",
                                        fate_probability_key = "fate_probabilities",
                                        save = "_fate_probs.png")
    atlas.pl.plot_tree(mudata = mudata,
                        embedding = "umap",
                        fate_probability_key = "fate_probabilities",
                        time_key = "pseudotime",
                        random_state = seed,
                        save = "_tree.png")

    print(results)

