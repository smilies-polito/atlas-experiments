import atlas
from muon import MuData


def _compute_results(mudata: MuData,
                    code: str,
                    seed:int = 42,
                    time_key: str = "pseudotime",
                    fate_key: str = "fate_probabilities") -> dict:

    results= {"code": None,
                "pearson_stat_KL": None,
                "pearson_pval_KL": None,
                "pearson_ciL_KL": None,
                "pearson_ciH_KL": None,
                "pearson_stat_SHE": None,
                "pearson_pval_SHE": None,
                "pearson_ciL_SHE": None,
                "pearson_ciH_SHE": None,
                "spearman_stat_KL": None,
                "spearman_pval_KL": None,
                "spearman_ciL_KL": None,
                "spearman_ciH_KL": None,
                "spearman_stat_SHE": None,
                "spearman_pval_SHE": None,
                "spearman_ciL_SHE": None,
                "spearman_ciH_SHE": None,
                "fate_index_stat": None,
                "fate_index_pval": None,
                "fate_index_ciL": None,
                "fate_index_ciH": None,
                "pseudotime_enrichment": None,
                "silhouette_soft": None,
                "silhouette_hard": None,
                }
    results["code"] = code
    stat, pval, ci = atlas.tl.pearson_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "kl_divergence",
                                                seed = seed)
    results["pearson_stat_KL"] = stat
    results["pearson_pval_KL"] = pval
    results["pearson_ciL_KL"] = ci[0]
    results["pearson_ciH_KL"] = ci[1]

    stat, pval, ci = atlas.tl.pearson_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "shannon_entropy",
                                                seed = seed)
    results["pearson_stat_SHE"] = stat
    results["pearson_pval_SHE"] = pval
    results["pearson_ciL_SHE"] = ci[0]
    results["pearson_ciH_SHE"] = ci[1]

    stat, pval, ci = atlas.tl.spearman_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "kl_divergence",
                                                seed = seed)
    results["spearman_stat_KL"] = stat
    results["spearman_pval_KL"] = pval
    results["spearman_ciL_KL"] = ci[0]
    results["spearman_ciH_KL"] = ci[1]

    stat, pval, ci = atlas.tl.spearman_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "shannon_entropy",
                                                seed = seed)
    results["spearman_stat_SHE"] = stat
    results["spearman_pval_SHE"] = pval
    results["spearman_ciL_SHE"] = ci[0]
    results["spearman_ciH_SHE"] = ci[1]

    stat, pval, ci, _ = atlas.tl.fate_concentration_index(mudata=mudata,
                                                        time_key = time_key,
                                                        fate_key = fate_key,
                                                        seed = seed)
    results["fate_index_stat"] = stat
    results["fate_index_pval"] = pval
    results["fate_index_ciL"] = ci[0]
    results["fate_index_ciH"] = ci[1]

    results["pseudotime_enrichment"] = atlas.tl.terminal_pseudotime_enrichment(
                                                            mudata = mudata,
                                                            time_key = time_key,
                                                            rank = True)

    results["silhouette_soft"] = atlas.tl.terminal_state_silhouette(mudata = mudata,
                                                                    fate_key = fate_key,
                                                                    soft_assignment= True)
    results["silhouette_hard"] = atlas.tl.terminal_state_silhouette(mudata = mudata,
                                                                    fate_key = fate_key,
                                                                    soft_assignment = False,
                                                                    time_key = time_key)
    return results


def _get_plots(mudata: MuData,
                code: str, 
                time_key: str = "pseudotime",
                embedding_key: str = "umap",
                fate_key: str = "fate_probabilities",
                seed: int = 42,
                ti_strategy: str = "palantir"):

    ek = "X_" + embedding_key 
    save = f"_{time_key}_{code}_{ti_strategy}.png"
    atlas.pl.plot_embedding(mudata = mudata,
                            embedding_key = ek,
                            observation = time_key,
                            save = save)

    save = f"_SHE_{code}_{ti_strategy}.png"
    atlas.pl.plot_embedding(mudata = mudata,
                            embedding_key = ek,
                            observation = "shannon_entropy",
                            save = save)

    save = f"_KL_{code}_{ti_strategy}.png"
    atlas.pl.plot_embedding(mudata = mudata,
                            embedding_key = ek,
                            observation = "kl_divergence",
                            save = save)
   
    save = f"_fates_{code}_{ti_strategy}.png"
    atlas.pl.plot_fate_probabilities(mudata = mudata,
                                        embedding_key = ek,
                                        fate_probability_key = fate_key,
                                        save = save)

    save = f"_tree_{code}_{ti_strategy}.png"
    try: 
        atlas.pl.plot_tree(mudata = mudata,
                            embedding = embedding,
                            fate_probability_key = fate_key,
                            time_key = time_key,
                            random_state = seed,
                            save = save)
    except Exception as e:
        print(ti_strategy, code, e)



