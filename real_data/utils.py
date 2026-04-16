import atlas
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from muon import MuData
from anndata import AnnData


def _compute_outlier(adata: AnnData, 
                metric: str,
                nmads: int):
    M = adata.obs[metric]
    outlier = (
                        (M < np.median(M) - nmads * median_abs_deviation(M)) | (
                         np.median(M) + nmads * median_abs_deviation(M) < M)
                )
    return outlier


def compute_entropy(data: MuData, fate_prob_key:str="palantir_fate_probabilities"):
    """Function mimicking ATLAS entropy computation"""
    def _minmax(x:np.ndarray) -> np.ndarray:
        if np.max(x) == np.min(x):
            return np.zeros_like(x)
        return (x- np.min(x)) / (np.max(x) - np.min(x))

    probs = data.obsm.get(fate_prob_key, None)
    if probs is None:
        raise ValueError("Compute fate probabilities before running entropy")

    if not isinstance(probs, pd.DataFrame):
        raise ValueError("Fate probabilities not a DataFrame")

    if (probs.shape[1] == 0
        or np.any(probs.sum(axis=1) == 0)
        or np.any(probs.sum(axis=0) == 0)):
        warnings.warn("No terminal states or cells with no developmental probability or state without assignment")
        data.obs["shannon_entropy"] = np.nan
        data.obs["kl_divergence"] = np.nan
        return
    shannon_entropy = scipy.stats.entropy(probs, axis=1)
    average_distribution = np.mean(probs, axis=0)
    kl_divergence = np.nan_to_num(scipy.stats.entropy(probs, average_distribution, axis=1, base=2),
                            nan=1.0,
                            copy=False)
    shannon_entropy, kl_divergence = _minmax(shannon_entropy), _minmax(kl_divergence)
    data.obs["shannon_entropy"] = pd.Series(shannon_entropy, index = data.obs.index)
    data.obs["kl_divergence"] = pd.Series(kl_divergence, index = data.obs.index)

def  _create_ts_dict(data:AnnData, fate_prob_key:str, cluster_key: str) -> dict:
    """Function creating terminal and initial states dicts from fate probabilities"""
    if fate_prob_key not in data.obsm or data.obsm[fate_prob_key] is None or data.obsm[fate_prob_key].shape[1] == 0:
        return {}
    terminal_cells = data.obsm[fate_prob_key].columns
    clusters = {}
    for cell in terminal_cells: 
        cluster = data.obs.loc[cell, cluster_key]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(cell)
    return clusters

def _invert_assignment(assignment):
    """Utility function for creating terminal/initial states from CellRank output"""
    if not isinstance(assignment.dtype, pd.CategoricalDtype):
        assignment = assignment.astype("category")

    inverted_assignment= { state: assignment.index[assignment==state].tolist()
                                for state in assignment.cat.categories}
    return inverted_assignment

def _assign_state_colors(mudata: MuData, cmap: str = "tab20"):
    """Function that assigns colors to the terminal/initial states"""
    all_states = set()
    if "fate_state_colors" not in mudata.uns:
        mudata.uns["fate_state_colors"] = {}
    color_map = mudata.uns["fate_state_colors"]

    for key in ["terminal_states", "initial_states", "intermediate_states"]:
        states = mudata.uns.get(key, None)
        if isinstance(states, dict):
            all_states.update(states.keys())

    new_states = [s for s in all_states if s not in color_map]
    if not new_states:
        return

    base_colors = plt.get_cmap(cmap).colors
    for state in sorted(new_states):
        idx = len(color_map)
        color = base_colors[idx % len(base_colors)]
        color_map[state] = to_hex(color)


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
    if ci is not None:
        results["pearson_ciL_KL"] = ci[0]
        results["pearson_ciH_KL"] = ci[1]
    else: 
        results["pearson_ciL_KL"] = None
        results["pearson_ciH_KL"] = None
        

    stat, pval, ci = atlas.tl.pearson_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "shannon_entropy",
                                                seed = seed)
    results["pearson_stat_SHE"] = stat
    results["pearson_pval_SHE"] = pval
    if ci is not None:
        results["pearson_ciL_SHE"] = ci[0]
        results["pearson_ciH_SHE"] = ci[1]
    else:
        results["pearson_ciL_SHE"] = None
        results["pearson_ciH_SHE"] = None

    stat, pval, ci = atlas.tl.spearman_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "kl_divergence",
                                                seed = seed)
    results["spearman_stat_KL"] = stat
    results["spearman_pval_KL"] = pval
    if ci is not None:
        results["spearman_ciL_KL"] = ci[0]
        results["spearman_ciH_KL"] = ci[1]
    else:
        results["spearman_ciL_KL"] = None
        results["spearman_ciH_KL"] = None

    stat, pval, ci = atlas.tl.spearman_correlation(mudata = mudata,
                                                key1 = time_key,
                                                key2 = "shannon_entropy",
                                                seed = seed)
    results["spearman_stat_SHE"] = stat
    results["spearman_pval_SHE"] = pval
    if ci is not None:
        results["spearman_ciL_SHE"] = ci[0]
        results["spearman_ciH_SHE"] = ci[1]
    else:
        results["spearman_ciL_SHE"] = None
        results["spearman_ciH_SHE"] = None

    stat, pval, ci, _ = atlas.tl.fate_concentration_index(mudata=mudata,
                                                        time_key = time_key,
                                                        fate_key = fate_key,
                                                        seed = seed)
    results["fate_index_stat"] = stat
    results["fate_index_pval"] = pval
    if ci is not None:
        results["fate_index_ciL"] = ci[0]
        results["fate_index_ciH"] = ci[1]
    else:
        results["fate_index_ciL"] = None
        results["fate_index_ciH"] = None

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
                            embedding = embedding_key,
                            fate_probability_key = fate_key,
                            time_key = time_key,
                            random_state = seed,
                            save = save)
    except Exception as e:
        print(ti_strategy, code, e)



