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
from .utils import initial_macrostate, terminal_macrostate, truth_like_fates, TERM_DICT, POTENCY_DICT
from atlas import pearson_correlation, spearman_correlation, kendall_correlation, fate_concentration_index, terminal_state_silhouette, terminal_pseudotime_enrichment_score, js_distance, terminal_state_score

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

def _save_simulation(data: AnnData, 
					tree: str,
					rd: float, 	
					sigma: float,
					knn_rna: int, 
					fixed_terminal: bool, 
					results: dict,
					ground_truth: dict,
					saving_folder: str,
					resources: dict):
	'''
		Save simulation results.
	'''
	code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}.h5ad"
	data.X = None
	data.obsm["true_fates"] = ground_truth["fate_probabilities"]
	data.uns["true_states"] = ground_truth["true_states"]
	data.uns["simulation_results"] = results
	data.write(os.path.join(saving_folder, code))

	results_path = os.path.join(saving_folder, "results.csv")
	resources_path = os.path.join(saving_folder, "resources.csv")
	resources["code"] = code

	with open(results_path, "a") as f:
		fcntl.flock(f, fcntl.LOCK_EX)
		pd.DataFrame([results]).to_csv(f, index=False, header=f.tell() == 0)
		fcntl.flock(f, fcntl.LOCK_UN)

	with open(resources_path, "a") as f:
		fcntl.flock(f, fcntl.LOCK_EX)
		pd.DataFrame([resources]).to_csv(f, index=False, header=f.tell() == 0)
		fcntl.flock(f, fcntl.LOCK_UN)
		
def _apply_metrics_and_visualize(data: AnnData,
								tree: str,
								rd: float, 
								sigma: float,
								knn_rna: int, 
								ts_dict: dict,
								terminal_clusters: list,
								true_probabilities: pd.DataFrame,
								failed: bool = False,
								fixed_terminal: bool = False,
								):
	'''
		Compute metrics.
	'''
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
					
	results["n_terminal_states"] = data.obsm["fate_probabilities"].shape[1]

	# no fate probabilities	
	if results["n_terminal_states"] <= 0:
		return results	

	#SUPERVISED - JSD
	if fixed_terminal:
		jsd = js_distance(data.obsm["fate_probabilities"], true_probabilities)
		if not jsd.index.equals(data.obs["pop"].index):
			jsd = jsd.loc[data.obs["pop"].index]
		jsdf = pd.DataFrame({"jsd": jsd, "cluster": data.obs["pop"]})
		jsdf["potency"] = jsdf["cluster"].map(POTENCY_DICT[tree])
		mean_jsd = jsdf.groupby("potency")["jsd"].mean()
        for cat in ["multipotent", "totipotent", "committed"]:
            results[f"jsd_{cat}"] = mean_jsd.get(cat, np.nan)

	#SUPERVISED - TERMINAL STATE SCORE
	tts, ttp, tsr, ttc, overall = terminal_state_score(data.obs["pseudotime"], data.obs["pop"], ts_dict, terminal_clusters)
	results["tts"] = tts
	results["ttp"] = ttp
	results["tsr"] = tsr
	results["ttc"] = ttc
	results["temporal_state_score"] = overall

	#UNSUPERVISED
	stat, pval, _ = spearman_correlation(data.obs["pseudotime"], data.obs["kl_divergence"], seed = 42)	
	results["spearman_stat_KLD"] = stat
	results["spearman_pval_KLD"] = pval
	stat, pval, _ = spearman_correlation(data.obs["pseudotime"], data.obs["shannon_entropy"], seed = 42)	
	results["spearman_stat_SHE"] = stat
	results["spearman_pval_SHE"] = pval
	stat, pval, _ = pearson_correlation(data.obs["pseudotime"], data.obs["kl_divergence"], seed = 42)	
	results["pearson_stat_KLD"] = stat
	results["pearson_pval_KLD"] = pval
	stat, pval, _ = pearson_correlation(data.obs["pseudotime"], data.obs["shannon_entropy"], seed = 42)	
	results["pearson_stat_SHE"] = stat
	results["pearson_pval_SHE"] = pval
	stat, pval, _, __ = fate_concentration_index(data.obsm["fate_probabilities"], data.obs["pseudotime"], seed = 42)
	results["fate_index_stat"] = stat
	results["fate_index_pval"] = pval
	results["terminal_silhouette_soft"] = terminal_state_silhouette(data.obsm["fate_probabilities"], soft_assignment=True)
	results["terminal_silhouette_pse"] = terminal_state_silhouette(data.obsm["fate_probabilities"], soft_assignment=False, pseudotime=data.obs["pseudotime"])
	results["terminal_enrichment"] = terminal_pseudotime_enrichment_score(terminal_states=ts_dict, pseudotime = data.obs["pseudotime"], rank=True)

	return results

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
		start_i_wall, start_i_cpu = time.perf_counter(), time.process_time()
		tracemalloc.start()
		kernel = cellrank.kernels.PseudotimeKernel(data,
											time_key = "pseudotime",
											connectivity_key = "connectivities")
		kernel.compute_transition_matrix(threshold_scheme = "hard", n_jobs = threads)
		_, init_mem_peak = tracemalloc.get_traced_memory()
		tracemalloc.stop()
		init_mem_peak = init_mem_peak / (1024 * 1024)  # bytes -> MiB
		end_i_wall, end_i_cpu = time.perf_counter(), time.process_time()
		init_wall, init_cpu = end_i_wall - start_i_wall, end_i_cpu - start_i_cpu
		failed = False
	except Exception as e:
		if tracemalloc.is_tracing():
			tracemalloc.stop()
		init_wall, init_cpu, init_mem_peak = None, None, None
		failed = True
	
	try:
		start_r_wall, start_r_cpu = time.perf_counter(), time.process_time()
		tracemalloc.start()
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
		_, run_mem_peak = tracemalloc.get_traced_memory()
		tracemalloc.stop()
		run_mem_peak = run_mem_peak / (1024 * 1024)  # bytes -> MiB
		end_r_wall, end_r_cpu = time.perf_counter(), time.process_time()
		run_wall, run_cpu = end_r_wall - start_r_wall, end_r_cpu - start_r_cpu
		failed = False
	except:
		if tracemalloc.is_tracing():
			tracemalloc.stop()
		failed = True
		run_wall, run_cpu, run_mem_peak = None, None, None

	resources = {	"run_wall_time": run_wall,
					"run_cpu_time": run_cpu,
					"run_mem_peak": run_mem_peak,
					"init_wall_time": init_wall,
					"init_cpu_time": init_cpu,
					"init_mem_peak": init_mem_peak
				}
	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics_and_visualize(data = data, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
					knn_rna = knn_rna,
					failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree],
					ts_dict = ts_dict, true_probabilities = true_fates)
	_save_simulation(data = data.copy(), tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
					knn_rna = knn_rna, fixed_terminal = False, 
					saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary, resources=resources)

	try:
		start_r_wall, start_r_cpu = time.perf_counter(), time.process_time()
		tracemalloc.start()
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
		_, run_mem_peak = tracemalloc.get_traced_memory()
		tracemalloc.stop()
		run_mem_peak = run_mem_peak / (1024 * 1024)  # bytes -> MiB
		end_r_wall, end_r_cpu = time.perf_counter(), time.process_time()
		run_wall, run_cpu = end_r_wall - start_r_wall, end_r_cpu - start_r_cpu
		failed = False
	except:
		if tracemalloc.is_tracing():
			tracemalloc.stop()
		failed = True
		run_wall, run_cpu, run_mem_peak = None, None, None

	resources = {	"run_wall_time": run_wall,
					"run_cpu_time": run_cpu,
					"run_mem_peak": run_mem_peak,
					"init_wall_time": init_wall,
					"init_cpu_time": init_cpu,
					"init_mem_peak": init_mem_peak
				}
	
	ts_dict = data.uns.get("terminal_states", {})
	results = _apply_metrics_and_visualize(data = data, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
					knn_rna = knn_rna,
					failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree],
					ts_dict = ts_dict, true_probabilities = true_fates)
	_save_simulation(data = data, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
					knn_rna = knn_rna, fixed_terminal = True, 
					saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary, resources=resources)


		
		


