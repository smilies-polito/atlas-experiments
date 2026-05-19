import os
import atlas
import fcntl
import muon as mu
import numpy as np
import pandas as pd
from muon import MuData
from anndata import AnnData
from typing import Literal, Union
from itertools import product
from .supervised_metrics import terminal_state_score, js_distance, kendall_correlation 

POTENCY_DICT = {"three_branches": {"4_1": "committed",
								"5_2": "committed",
								"5_3": "committed",
								"4_5": "totipotent"},
				"five_branches": {"6_7": "totipotent",
								"6_1": "committed",
								"7_9": "multipotent",	
								"7_8": "multipotent",
								"8_2": "committed",
								"8_3": "committed",
								"9_4": "committed",
								"9_5": "committed"}
				}

TERM_DICT = {"three_branches": ["4_1", "5_2", "5_3"],
			"five_branches": ["9_4", "9_5", "8_2", "8_3", "6_1"]}

PHYLA5 = pd.DataFrame([[1,0,0,0,0],[0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0],
			 [0,0,0,0,1], [1,1,1,1,1], [0,1,1,0,0], [0,0,0,1,1]],
			columns=["6_1", "8_2", "8_3", "9_4", "9_5"],
			index=["6_1", "8_2", "8_3", "9_4", "9_5", "6_7", "7_8", "7_9"])


PHYLA3 = pd.DataFrame([[1, 1, 1], [1,0,0], [0,1,0], [0,0,1]],
			index = ["4_5", "4_1", "5_2", "5_3"],
			columns = ["4_1", "5_2", "5_3"])

DEV_DICT = {"three_branches": PHYLA3, 
		"five_branches": PHYLA5}

				
def truth_like_fates(pseudotime: pd.Series, 
			membership: pd.Series, 
			tree:Literal["three_branches", "five_branches"]="three_branches",
			alpha: float = 2.0)-> pd.DataFrame:

	development = DEV_DICT[tree]
	accessibility = development.loc[membership.values].values
	W = pd.DataFrame(accessibility,
			index = pseudotime.index,
			columns = development.columns)
	row_sum = W.sum(axis=1)
	if np.any(row_sum==0):
		raise ValueError("Some cells have no accessible terminal states")
	return  W.divide(row_sum, axis=0)
		

def initial_macrostate(mudata:Union[AnnData, MuData], pseudotime_key: str, n_cells: int) -> list:
	pseudotime = mudata.obs[pseudotime_key]
	return pseudotime.nsmallest(n_cells).index.tolist()

def terminal_macrostate(mudata: Union[AnnData, MuData],
			pseudotime_key: str,
			cluster_key: str,
			terminal_state: str,
			n_cells: int) -> list:
	pseudotime = mudata[mudata.obs[cluster_key] == terminal_state].obs[pseudotime_key]
	return pseudotime.nlargest(n_cells).index.tolist()


def _save_simulation(
					mudata: MuData,
					tree: str,
					rd: float,	 
					sigma: float,
					knn_rna: int, 
					knn_activity: int,
					wnn: None | int,
					fixed_terminal: bool, 
					results: dict,
					ground_truth: dict,
					saving_folder: str):

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
	mudata.uns["simulation_results"] = results

	if "DM_EigenVectors" in mudata.obsm:
		mudata.obsm["DM_EigenVectors"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors"].columns]
	if "DM_EigenVectors_multiscaled" in mudata.obsm:
		mudata.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in mudata.obsm["DM_EigenVectors_multiscaled"].columns]
	mudata.write(os.path.join(saving_folder, f"{code}.h5mu"))

	results_path = os.path.join(saving_folder, "results.csv")
	with open(results_path, "a") as f:
		fcntl.flock(f, fcntl.LOCK_EX)
		pd.DataFrame([results]).to_csv(f, index=False, header=f.tell() == 0)
		fcntl.flock(f, fcntl.LOCK_UN)


def _construct_mudata(mudata: AnnData,
					fate_key: str) -> MuData:
	new_ann = MuData({"rna": mudata})
	new_ann.obsm[fate_key] = mudata.obsm[fate_key]
	new_ann.uns["terminal_states"] = mudata.uns["terminal_states"]
	new_ann.uns["initial_states"] = mudata.uns["initial_states"]
	if "intermediate_states" in mudata.uns.keys():
		new_ann.uns["intermediate_states"] = mudata.uns["intermediate_states"]
	else:
		new_ann.uns["intermediate_states"] = {}
	return new_ann



def _apply_metrics(mudata: Union[MuData, AnnData],
								tree: str,
								rd: float, 
								sigma: float,
								knn_rna: int, 
								ts_dict: dict,
								terminal_clusters: list,
								true_probabilities: pd.DataFrame,
								knn_activity: None | int = None,
								wnn: None | int = None,
								failed: bool = False,
								fixed_terminal: bool = False,
								pseudotime_key: str = "pseudotime", 
								true_pseudotime_key: str = "rna:pseudotime",
								fate_key: str = "fate_probabilities",
								cluster_key: str = "rna:pop",
								kl_div_key: str = "kl_divergence",
								entropy_key: str = "shannon_entropy",
								ground_truth_pseudotime: bool = True,
								seed: int = 42
								):
	if knn_activity is not None and wnn is not None:
		code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}:{knn_activity}:{wnn}"
	else:
		code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}"

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

	if isinstance(mudata, AnnData):
		mudata = _construct_mudata(mudata, fate_key = fate_key )
					
	results["n_terminal_states"] = mudata.obsm[fate_key].shape[1]

	# SUPERVISED - correlation true and inferred pseudotime
	if ground_truth_pseudotime:
		stat, pval, _ = atlas.tl.spearman_correlation(mudata = mudata, 
														key1 = pseudotime_key,
														key2 = true_pseudotime_key,
														seed = seed
													 )
		results["spearman_stat_pseudotime"] = stat
		results["spearman_pval_pseudotime"] = pval
		stat, pval =  kendall_correlation(mudata = mudata,
											pred_time = pseudotime_key,
											true_time = true_pseudotime_key)
		results["kendall_stat_pseudotime"] = stat
		results["kendall_pval_pseudotime"] = pval

	if results["n_terminal_states"] <= 0:
		return results	

	#SUPERVISED - JSD
	if fixed_terminal:
		jsd = js_distance(mudata = mudata,
							truth = true_probabilities,
							fate_key = fate_key
						)
							
		if not jsd.index.equals(mudata.obs[cluster_key].index):
			jsd = jsd.loc[mudata.obs[cluster_key].index]
		jsdf = pd.DataFrame({"jsd": jsd, "cluster": mudata.obs[cluster_key]})
		jsdf["potency"] = jsdf["cluster"].map(POTENCY_DICT[tree])
		mean_jsd = jsdf.groupby("potency")["jsd"].mean()
		for cat in ["multipotent", "totipotent", "committed"]:
			results[f"jsd_{cat}"] = mean_jsd.get(cat, np.nan)

	#SUPERVISED - TERMINAL STATE SCORE
	tts, ttp, tsr, ttc, overall = terminal_state_score(mudata = mudata,
														time_key = pseudotime_key,
														cluster_key = cluster_key,
														terminal_clusters = terminal_clusters)
	results["tts"] = tts
	results["ttp"] = ttp
	results["tsr"] = tsr
	results["ttc"] = ttc
	results["temporal_state_score"] = overall

	#UNSUPERVISED
	stat, pval, _ = atlas.tl.spearman_correlation(
			mudata = mudata,
			key1 = pseudotime_key,
			key2 = kl_div_key,
			seed = seed)
	results["spearman_stat_KLD"] = stat
	results["spearman_pval_KLD"] = pval
	stat, pval, _ = atlas.tl.spearman_correlation(
			mudata = mudata,
			key1 = pseudotime_key,
			key2 = entropy_key,
			seed = seed)
	results["spearman_stat_SHE"] = stat
	results["spearman_pval_SHE"] = pval
	stat, pval, _ = atlas.tl.pearson_correlation(
			mudata = mudata,
			key1 = pseudotime_key,
			key2 = kl_div_key,
			seed = seed)
	results["pearson_stat_KLD"] = stat
	results["pearson_pval_KLD"] = pval
	stat, pval, _ = atlas.tl.pearson_correlation(
			mudata = mudata,
			key1 = pseudotime_key,
			key2 = entropy_key,
			seed = seed)
	results["pearson_stat_SHE"] = stat
	results["pearson_pval_SHE"] = pval

	stat, pval, _, __ = atlas.tl.fate_concentration_index(
			mudata = mudata,
			fate_key = fate_key,
			time_key = pseudotime_key,
			seed = seed)
	results["fate_index_stat"] = stat
	results["fate_index_pval"] = pval

	results["terminal_silhouette_soft"] = atlas.tl.terminal_state_silhouette(
			mudata = mudata,
			fate_key = fate_key,
			soft_assignment = True)
	results["terminal_silhouette_pse"] = atlas.tl.terminal_state_silhouette(
			mudata = mudata,
			fate_key = fate_key,
			time_key = pseudotime_key,
			soft_assignment=False)

	results["terminal_enrichment"] = atlas.tl.terminal_pseudotime_enrichment(
											mudata = mudata,
											time_key = pseudotime_key,
											rank = True)
	return results


def _save_simulation_rna(data: AnnData, 
					tree: str,
					rd: float, 	
					sigma: float,
					knn_rna: int, 
					fixed_terminal: bool, 
					results: dict,
					ground_truth: dict,
					saving_folder: str):

	code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}.h5ad"
	data.X = None
	data.obsm["true_fates"] = ground_truth["fate_probabilities"]
	data.uns["true_states"] = ground_truth["true_states"]

	if "DM_EigenVectors" in data.obsm:
		data.obsm["DM_EigenVectors"].columns = [str(c) for c in data.obsm["DM_EigenVectors"].columns]
	if "DM_EigenVectors_multiscaled" in data.obsm:
		data.obsm["DM_EigenVectors_multiscaled"].columns = [str(c) for c in data.obsm["DM_EigenVectors_multiscaled"].columns]

	data.uns["simulation_results"] = results
	data.write(os.path.join(saving_folder, code))

	results_path = os.path.join(saving_folder, "results.csv")

	with open(results_path, "a") as f:
		fcntl.flock(f, fcntl.LOCK_EX)
		pd.DataFrame([results]).to_csv(f, index=False, header=f.tell() == 0)
		fcntl.flock(f, fcntl.LOCK_UN)


