import os
import argparse
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
from atlas import ATLAS, pearson_correlation, spearman_correlation, kendall_correlation, fate_concentration_index, terminal_state_silhouette, terminal_pseudotime_enrichment_score, js_distance, terminal_state_score
from muon import MuData
from anndata import AnnData
from itertools import product
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT


def _save_simulation(atlas:ATLAS, 
					tree: str,
					rd: float, 	
					sigma: float,
					knn_rna: int, 
					knn_activity: int,
					wnn: int,
					fixed_terminal: bool, 
					results: dict,
					saving_folder: str):

	code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}:{knn_activity}:{wnn}.h5mu"
	data = atlas.get_data()
	mod = AnnData( X = csr_matrix( np.zeros(shape = (data.n_obs, len(data["rna"].var_names))) ),
					obs = pd.DataFrame([], index = data["rna"].obs.index) ,
					var = pd.DataFrame([], index = data["rna"].var.index) )
	mudata = MuData({"sim": mod})
	mudata.obsm["fate_probabilities"] = data.obsm["fate_probabilities"]
	mudata.obsm["true_probabilities"] = data.obsm["true_probabilities"]
	mudata.obsm["X_umap"] = data.obsm["X_umap"]
	mudata.obsp["wnn_connectivities"] = data.obsp["wnn_connectivities"] 
	mudata.obsp["wnn_distances"] = data.obsp["wnn_distances"] 
	mudata.uns = data.uns.copy()
	mudata.uns["simulation_results"] = results
	
	mudata.write(os.path.join(saving_folder, code))
			

def _apply_metrics_and_visualize(atlas: ATLAS,
								tree: str,
								rd: float, 
								sigma: float,
								knn_rna: int, 
								knn_activity: int,
								wnn: int,
								ts_dict: dict,
								terminal_clusters: list,
								failed: bool = False,
								fixed_terminal: bool = False,
								):
	code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}:{knn_activity}:{wnn}"
	results = {
				"code" : code,
				"failed": failed,
				"n_inferred_terminal_states" : None,
				"fixed_terminal" : fixed_terminal,
				"spearman_stat_pseudotime": None,
				"spearman_pval_pseudotime": None,
				"kendall_stat_pseudotime": None,
				"kendall_pval_pseudotime": None,
				"jsd": None,
				"tts": None,
				"ttp": None,
				"ttc": None,
				"temporal_state_score": None,
				"pearson_stat_KLD": None,
				"pearson_pval_KLD": None,
				"pearson_stat_SHE": None,
				"pearson_pval_SHE": None,
				"spearman_stat_KLD": None,
				"spearman_pval_KLD": None,
				"spearman_stat_SHE": None,
				"spearman_pval_SHE": None,
				"fate_index_stat": None,
				"fate_index_pval": None,
				"terminal_silhouette_soft": None,
				"terminal_silhouette_pse": None,
			}

	mudata = atlas.get_data()
	if failed:
		return results

	if "fate_probabilities" in mudata.obsm and mudata.obsm["fate_probabilities"] is not None:
		results["n_terminal_states"] = mudata.obsm["fate_probabilities"].shape[1]

	# SUPERVISED 
	stat, pval, _ = spearman_correlation(data.obs["rna:pseudotime"], data.obs["pseudotime"], seed = atlas.random_state)	
	results["spearman_stat_pseudotime"] = stat
	results["spearman_pval_pseudotime"] = pval
	stat, pval =  kendall_correlation(data.obs["rna:pseudotime"], data.obs["pseudotime"])
	results["kendall_stat_pseudotime"] = stat
	results["kendall_pval_pseudotime"] = pval

	if fixed_terminal:
		jsd = js_distance(data.obsm["fate_probabilities"], data.obsm["true_probabilities"])
		if not jsd.index.equals(data.obs["rna:pop"].index):
			jsd = jsd.loc[data.obs["rna:pop"].index]
		jsdf = pd.DataFrame({"jsd": jsd, "cluster": data.obs["rna:pop"]})
		jsdf["potency"] = jsdf["cluster"].map(POTENCY_DICT[tree])
		results["jsd"] = jsdf.groupby("potency")["jsd"].mean()
		

	if not fixed_terminal:
		tts, ttp, ttc, overall = terminal_state_score(data.obs["rna:pseudotime"], data.obs["rna:pop"], ts_dict, terminal_clusters)
		results["tts"] = tts
		results["ttp"] = ttp
		results["ttc"] = ttc
		results["temporal_state_score"] = overall

	# UNSUPERVISED
	stat, pval, _ = spearman_correlation(data.obs["pseudotime"], data.obs["kl_divergence"], seed = atlas.random_state)	
	results["spearman_stat_KLD"] = stat
	results["spearman_pval_KLD"] = pval
	stat, pval, _ = spearman_correlation(data.obs["pseudotime"], data.obs["shannon_entropy"], seed = atlas.random_state)	
	results["spearman_stat_SHE"] = stat
	results["spearman_pval_SHE"] = pval
	stat, pval, _ = pearson_correlation(data.obs["pseudotime"], data.obs["kl_divergence"], seed = atlas.random_state)	
	results["pearson_stat_KLD"] = stat
	results["pearson_pval_KLD"] = pval
	stat, pval, _ = pearson_correlation(data.obs["pseudotime"], data.obs["shannon_entropy"], seed = atlas.random_state)	
	results["pearson_stat_SHE"] = stat
	results["pearson_pval_SHE"] = pval
	stat, pval, _, __ = fate_concentration_index(data.obsm["fate_probabilities"], data.obs["pseudotime"], seed = atlas.random_state)
	results["fate_index_stat"] = stat
	results["fate_index_pval"] = pval
	results["terminal_silhouette_soft"] = terminal_state_silhouette(data.obsm["fate_probabilities"], soft_assignment=True)
	results["terminal_silhouette_pse"] = terminal_state_silhouette(data.obsm["fate_probabilities"], soft_assignment=False, pseudotime=data.obs["pseudotime"])
	
	# VISUALIZATION
	# pseudotime
	atlas.plot_embedding(embedding_key = "X_umap",
						observation = "pseudotime",
						save = f"_PSTIME{code}.png",
						show= False)
	# shannon entropy	
	atlas.plot_embedding(embedding_key = "X_umap",
						observation = "kl_divergence",
						save = f"_KLDIV_{code}.png",
						show= False)
	# kl divergence
	atlas.plot_embedding(embedding_key = "X_umap",
						observation = "shannon_entropy",
						save = f"_SHENTR_{code}.png",
						show= False)
	# fate probabilities
	atlas.plot_fate_probabilities(embedding_key= "X_umap",
									states = None,
									save =  "_fates_{code}.png",
									show= False)
	# tree
	atlas.plot_tree(embedding_key = "umap",
					save = f"_{code}.png",
					color = "rna:pop", 
					color_milestones = False,
					show = False)

	return results


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)
	
	# parse arguments from 
	parser = argparse.ArgumentParser() 
	parser.add_argument("--tree", type=str)
	parser.add_argument("--rd", type=float)
	parser.add_argument("--sigma", type=float)
	parser.add_argument("--knn_rna", type=int)
	parser.add_argument("--knn_activity", type=int)
	parser.add_argument("--wnn", type=int)
	args = parser.parse_args()
	
	working_directory = os.getcwd() # set path to repository 
	data_path = os.path.join(working_directory, "data", "simulated_data", args.tree)
	saving_simulation_path = os.path.join(working_directory, "data", "simulated_data", "simulations", "palantir")
	n_pcs_rna = 20
	n_pcs_activity = 10
	
	# Lettura dei dati 
	diff_cif_fraction, cif_sigma = args.rd, args.sigma
	activity = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
	spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
	unspliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_unspliced.tsv"), sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

	# creazione matrice di attività
	activity = AnnData(X=csr_matrix(activity.values), 
			obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
			var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
	# creazione matrice di rna	
	total_rna = csr_matrix(spliced.values.T)
	rna = AnnData(X=csr_matrix(total_rna), 
			obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
			var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
		
	# creazione oggetto MuData 
	data = MuData({"rna":rna, "activity":activity})
	
	# rna preprocessing
	sc.pp.normalize_total(data["rna"])
	sc.pp.log1p(data["rna"])
	sc.pp.pca(data["rna"], random_state=seed, use_highly_variable=False)

	# activity preprocessing
	sc.pp.normalize_total(data["activity"])
	sc.pp.pca(data["activity"], random_state=seed, use_highly_variable=False)

	knn_rna, knn_activity, wnn = args.knn_rna, args.knn_activity, args.wnn
			
	atlas = ATLAS(mudata = data,
					method= "palantir",
					fragment_path = None,
					random_state =  seed)

	atlas.preprocessing(n_pcs_rna = n_pcs_rna,
						n_pcs_act = n_pcs_activity,
						knn_rna = knn_rna,
						knn_act = knn_activity,
						n_neighbors = wnn) 

	mu.pl.embedding(data, 
					basis="X_umap", 
					color=["rna:pop", "rna:pseudotime"], 
					show=False, 
					save = f"{diff_cif_fraction}_{cif_sigma}_{knn_rna}:{knn_activity}:{wnn}.png" )

	# ground truth 
	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "rna:pseudotime", 
									n_cells=1)
	terminal_cells = []
	expected_terminal_clusters = TERM_DICT[args.tree]

	for branch in expected_terminal_clusters:
		cell = terminal_macrostate(mudata = data,
											pseudotime_key = "rna:pseudotime", 
											cluster_key = "rna:pop",
											terminal_state = branch,
											n_cells=1)			
		terminal_cells.extend(cell)
	true_fates = truth_like_fates(pseudotime = data.obs["rna:pseudotime"],
									membership = data.obs["rna:pop"],
									tree = args.tree)
	data.obsm["true_probabilities"] = true_fates
	data.uns["selected_cells"] = {"initial" : early_cell,	
								"terminal": terminal_cells}

	# test non fixed terminal states	
	try: 
		atlas.run(early_cell = early_cell[0],
				cluster_key = "rna:pop",
				terminal_states = None,
				knn = 30,
				num_waypoints = 250)
		failed = False	
	except:
		failed = True

	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics_and_visualize(atlas = atlas, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
					knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
					failed = failed, fixed_terminal = False, terminal_clusters = expected_terminal_clusters,
					ts_dict = ts_dict)

	_save_simulation(atlas = atlas, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, knn_rna = knn_rna, knn_activity = 
					knn_activity, wnn= wnn, fixed_terminal = False, saving_folder = saving_simulation_path, results=results)

	# test fixed terminal states 
	try:
		atlas.run(early_cell = early_cell[0], 
					cluster_key = "rna:pop",
					terminal_states = terminal_cells,
					knn = 30,
					num_waypoints = 250)
		failed = False
	except:
		failed = True

	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics_and_visualize(atlas = atlas, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
					knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
					failed = failed, fixed_terminal = True, terminal_clusters = expected_terminal_clusters,
					ts_dict = ts_dict)

	_save_simulation(atlas = atlas, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, knn_rna = knn_rna, knn_activity = 
					knn_activity, wnn= wnn, fixed_terminal = True, saving_folder = saving_simulation_path, results = results)
