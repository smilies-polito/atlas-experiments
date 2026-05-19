import os
import json
import fcntl
import time
import warnings
import tracemalloc
import argparse
import palantir
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import scipy 
from muon import MuData
from anndata import AnnData
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation_rna, _apply_metrics

def compute_entropy(data: AnnData, fate_prob_key:str="palantir_fate_probabilities"):
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


if __name__=="__main__":
	seed = 42
	working_directory = os.getcwd() # set path to repository 
	np.random.seed(seed)

	parser = argparse.ArgumentParser() 
	parser.add_argument("--tree", type=str)
	parser.add_argument("--rd", type=float)
	parser.add_argument("--sigma", type=float)
	parser.add_argument("--knn_rna", type=int)
	args = parser.parse_args()
	
	# DATA CONTRUCTION 
	tree = args.tree
	diff_cif_fraction, cif_sigma = args.rd, args.sigma
	knn_rna= args.knn_rna
	n_pcs_rna = 20
	
	data_path = os.path.join(working_directory, "data", "simulated_data", tree)
	saving_simulation_path = os.path.join(working_directory, "output", "simulations", "palantir_rna")
	spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

	# create rna matrix
	data = AnnData(X=csr_matrix(spliced.values.T), 
			obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
			var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	data.obs = data.obs.merge(metadata, how="left", left_index=True, right_index=True)
	sc.pp.normalize_total(data)
	sc.pp.log1p(data)
	sc.pp.pca(data, random_state=seed, use_highly_variable=False)

	# ground truth construction
	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "pseudotime", 
									n_cells=1)
	terminal_cells = []
	for branch in TERM_DICT[tree]:
		cell = terminal_macrostate(mudata = data,
									pseudotime_key = "pseudotime", 
									cluster_key = "pop",
									terminal_state = branch,
									n_cells=1)			
		terminal_cells.extend(cell)

	true_fates = truth_like_fates(pseudotime = data.obs["pseudotime"],
									membership = data.obs["pop"],
									tree = args.tree)

	truth_dictionary = { "fate_probabilities" :true_fates, 
						"true_states": {"initial" : early_cell,	
										"terminal": terminal_cells}
						}
	# RUN WITH NO FIXED TERMINAL 
	try: 
		n_components, num_waypoints = 5, 250
		kernel = palantir.utils.compute_kernel(data, knn=knn_rna)
		diffusion_maps = palantir.utils.diffusion_maps_from_kernel(kernel = kernel, n_components=n_components, seed = seed)
		data.obsp["DM_Similarity"] = diffusion_maps["T"]
		data.obsm["DM_EigenVectors"] = diffusion_maps["EigenVectors"].set_index(data.obs.index)
		data.uns["DM_EigenValues"] = diffusion_maps["EigenValues"].values
		eigenvectors = pd.DataFrame(data.obsm["DM_EigenVectors"], index = data.osb_names) if not isinstance(data.obsm["DM_EigenVectors"], pd.DataFrame) else data.obsm["DM_EigenVectors"]
		dm_dict = {"EigenValues": data.uns["DM_EigenValues"], "EigenVectors": eigenvectors}
		result = palantir.utils.determine_multiscale_space(dm_res = dm_dict)
		data.obsm["DM_EigenVectors_multiscaled"] = result
		palantir.core.run_palantir(data=data,
						early_cell = early_cell,
						terminal_states = None,
						num_waypoints = num_waypoints,
						seed = seed)
		data.uns["terminal_states"] = _create_ts_dict(data, fate_prob_key="palantir_fate_probabilities", cluster_key="pop")
		data.uns["initial_states"] = {data.obs.loc[early_cell[0], "pop"]: early_cell}
		clusters =  data.obs.loc[data.obsm["palantir_fate_probabilities"].columns, "pop"].astype(str)
		data.obsm["palantir_fate_probabilities"].columns = clusters
		data.obsm["palantir_fate_probabilities"] = data.obsm["palantir_fate_probabilities"].groupby(level=0, axis=1).sum()
		compute_entropy(data = data, fate_prob_key="palantir_fate_probabilities")
		failed = False
	except Exception as e:
		print(e)
		failed = True

	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics(mudata = data, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma, knn_rna = knn_rna, pseudotime_key="rna:palantir_pseudotime", true_pseudotime_key = "rna:pseudotime", cluster_key = "rna:pop", failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree],  ts_dict = ts_dict, true_probabilities = true_fates, fate_key = "palantir_fate_probabilities", kl_div_key = "rna:kl_divergence", entropy_key="rna:shannon_entropy")
	_save_simulation_rna(data = data, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
					knn_rna = knn_rna, fixed_terminal = False, 
					saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)

	# RUN WITH FIXED TERMINAL
	try: 
		n_components, num_waypoints = 5, 250
		kernel = palantir.utils.compute_kernel(data, knn=knn_rna)
		diffusion_maps = palantir.utils.diffusion_maps_from_kernel(kernel = kernel, n_components=n_components, seed = seed)
		data.obsp["DM_Similarity"] = diffusion_maps["T"]
		data.obsm["DM_EigenVectors"] = diffusion_maps["EigenVectors"].set_index(data.obs.index)
		data.uns["DM_EigenValues"] = diffusion_maps["EigenValues"].values
		eigenvectors = pd.DataFrame(data.obsm["DM_EigenVectors"], index = data.osb_names) if not isinstance(data.obsm["DM_EigenVectors"], pd.DataFrame) else data.obsm["DM_EigenVectors"]
		dm_dict = {"EigenValues": data.uns["DM_EigenValues"], "EigenVectors": eigenvectors}
		result = palantir.utils.determine_multiscale_space(dm_res = dm_dict)
		data.obsm["DM_EigenVectors_multiscaled"] = result
		palantir.core.run_palantir(data=data,
						early_cell = early_cell,
						terminal_states = terminal_cells,
						num_waypoints = num_waypoints,
						seed = seed)
		data.uns["terminal_states"] = _create_ts_dict(data, fate_prob_key="palantir_fate_probabilities", cluster_key="pop")
		data.uns["initial_states"] = {data.obs.loc[early_cell[0], "pop"]: early_cell}
		clusters =  data.obs.loc[data.obsm["palantir_fate_probabilities"].columns, "pop"].astype(str)
		data.obsm["palantir_fate_probabilities"].columns = clusters
		data.obsm["palantir_fate_probabilities"] = data.obsm["palantir_fate_probabilities"].groupby(level=0, axis=1).sum()
		compute_entropy(data = data, fate_prob_key="palantir_fate_probabilities")
		failed = False
	except Exception as e:
		print(e)
		failed = True
			
	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics(mudata = data, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma, knn_rna = knn_rna, failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree], true_pseudotime_key="rna:pseudotime", fate_key = "palantir_fate_probabilities", pseudotime_key = "rna:palantir_pseudotime", cluster_key= "rna:pop", ts_dict = ts_dict, true_probabilities = true_fates, kl_div_key = "rna:kl_divergence", entropy_key= "rna:shannon_entropy")
	_save_simulation_rna(data = data, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, knn_rna = knn_rna, fixed_terminal = True, saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)
