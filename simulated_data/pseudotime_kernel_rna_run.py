import os
import time
import json
import fcntl
import scipy 
import warnings
import argparse
import tracemalloc
import cellrank 
import muon as mu 
import numpy as np
import pandas as pd
import scanpy as sc
from muon import MuData
from anndata import AnnData
from scipy.sparse import csr_matrix
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT, _save_simulation_rna, _apply_metrics

def _invert_assignment(assignment):
	if not isinstance(assignment.dtype, pd.CategoricalDtype):
		assignment = assignment.astype("category")

	inverted_assignment= { state: assignment.index[assignment==state].tolist()
								for state in assignment.cat.categories}
	return inverted_assignment


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

		
if __name__=="__main__":
	seed = 42
	threads = 3
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
	saving_simulation_path = os.path.join(working_directory, "output", "simulations", "pseudotime_kernel_rna")
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
	sc.pp.neighbors(data, n_pcs=n_pcs_rna, knn=knn_rna, random_state = seed)
	data.obs["pop"] = data.obs["pop"].astype("category")

	# ground truth construction
	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "pseudotime", 
									n_cells=30)
	early_cluster_subset = data.obs["pop"].loc[early_cell].value_counts().idxmax()
	initial_cells = {early_cluster_subset: early_cell}

	terminal_cells = {}
	for branch in TERM_DICT[tree]:
		cell = terminal_macrostate(mudata = data,
									pseudotime_key = "pseudotime", 
									cluster_key = "pop",
									terminal_state = branch,
									n_cells=30)			
		terminal_cells[branch] = cell

	true_fates = truth_like_fates(pseudotime = data.obs["pseudotime"],
									membership = data.obs["pop"],
									tree = tree)
	truth_dictionary = { "fate_probabilities": true_fates,	
							"true_states": {"initial": initial_cells,
											"terminal": terminal_cells}
						}
	try:
		kernel = cellrank.kernels.PseudotimeKernel(data,
											time_key = "pseudotime",
											connectivity_key = "connectivities")
		kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = threads)
		failed = False
	except Exception as e:
		print(e)
		failed = True
	
	try:
		g = cellrank.estimators.GPCCA(kernel)
		g.compute_schur()
		g.compute_macrostates(n_states=None, cluster_key = "pop")
		g.predict_terminal_states(allow_overlap=True)
		g.predict_initial_states(allow_overlap=True)
		g.compute_fate_probabilities(use_petsc=True, n_jobs=threads)
		data.obsm["fate_probabilities"] = pd.DataFrame(g.fate_probabilities.X,
												index = data.obs_names,
												columns = g.fate_probabilities.names)
		data.uns["initial_states"] = _invert_assignment(g.initial_states)
		data.uns["terminal_states"] = _invert_assignment(g.terminal_states)
		compute_entropy(data = data, fate_prob_key="fate_probabilities")
		failed = False
	except Exception as e:
		print(e)
		failed = True

	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics(mudata = data, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma, knn_rna = knn_rna, fate_key = "fate_probabilities", pseudotime_key="rna:pseudotime", entropy_key="rna:shannon_entropy", kl_div_key = "rna:kl_divergence", cluster_key="rna:pop",  ground_truth_pseudotime=False, failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree], ts_dict = ts_dict, true_probabilities = true_fates)
	_save_simulation_rna(data = data.copy(), tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, knn_rna = knn_rna, fixed_terminal = False, saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)

	try:
		kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = -1)
		g = cellrank.estimators.GPCCA(kernel)
		g.compute_schur()
		g.set_initial_states(initial_cells)
		g.set_terminal_states(terminal_cells)
		g.compute_fate_probabilities(use_petsc=True, n_jobs=threads)
		data.obsm["fate_probabilities"] = pd.DataFrame(g.fate_probabilities.X,
										index = data.obs_names,
										columns = g.fate_probabilities.names)
		data.uns["initial_states"] = _invert_assignment(g.initial_states)
		data.uns["terminal_states"] = _invert_assignment(g.terminal_states)
		compute_entropy(data = data, fate_prob_key="fate_probabilities")
		failed = False
	except Exception as e:
		print(e)
		failed = True

	
	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics(mudata = data, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma, knn_rna = knn_rna, fate_key = "fate_probabilities", entropy_key = "rna:shannon_entropy", kl_div_key = "rna:kl_divergence", cluster_key="rna:pop", pseudotime_key="rna:pseudotime", ground_truth_pseudotime=False, failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree], ts_dict = ts_dict, true_probabilities = true_fates)
	_save_simulation_rna(data = data, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, knn_rna = knn_rna, fixed_terminal = True, saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary)


		
		


