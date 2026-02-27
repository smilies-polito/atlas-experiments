import os
import json
import time
import fcntl
import tracemalloc
import argparse
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
from atlas import ATLAS, pearson_correlation, spearman_correlation, kendall_correlation, fate_concentration_index, terminal_state_silhouette, terminal_pseudotime_enrichment_score, js_distance, terminal_state_score
from muon import MuData
from anndata import AnnData
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
					ground_truth: dict,
					saving_folder: str,
					resources: dict):
	'''
		Save simulation results.
	'''
	code = f"{tree}_{fixed_terminal}_{rd}_{sigma}_{knn_rna}:{knn_activity}:{wnn}.h5mu"
	data = atlas.get_data().copy()
	del data.mod[atlas._impl.activity_key].obsm
	del data.mod[atlas._impl.activity_key].obsp
	del data.mod[atlas._impl.activity_key].uns
	del data.mod[atlas._impl.activity_key].varm
	data.mod[atlas._impl.activity_key].X = None
	del data.mod[atlas._impl.rna_key].obsm
	del data.mod[atlas._impl.rna_key].obsp
	del data.mod[atlas._impl.rna_key].uns
	del data.mod[atlas._impl.rna_key].varm
	data.mod[atlas._impl.activity_key].X = None

	data.obsm["true_fates"] = ground_truth["fate_probabilities"]
	data.uns["true_states"] = ground_truth["true_states"]

	if results["jsd"] is not None:
		results["jsd"] = {"values": results["jsd"].to_list(),
							"index": results["jsd"].index.to_list() }
	data.uns["simulation_results"] = results
	data.write(os.path.join(saving_folder, code))

	results_path = os.path.join(saving_folder, "results.csv")
	resources_path = os.path.join(saving_folder, "resources.csv")
	flat_results = { k: json.dumps(v) if isinstance(v, (dict, list)) else v	for k,v in results.items() }
	resources["code"] = code

	with open(results_path, "a") as f:
		fcntl.flock(f, fcntl.LOCK_EX)
		pd.DataFrame([flat_results]).to_csv(f, index=False, header=f.tell() == 0)
		fcntl.flock(f, fcntl.LOCK_UN)

	with open(resources_path, "a") as f:
		fcntl.flock(f, fcntl.LOCK_EX)
		pd.DataFrame([resources]).to_csv(f, index=False, header=f.tell() == 0)
		fcntl.flock(f, fcntl.LOCK_UN)


def _apply_metrics_and_visualize(atlas: ATLAS,
								tree: str,
								rd: float, 
								sigma: float,
								knn_rna: int, 
								knn_activity: int,
								wnn: int,
								ts_dict: dict,
								terminal_clusters: list,
								true_probabilities: pd.DataFrame,
								failed: bool = False,
								fixed_terminal: bool = False,
								):
	'''
		Compute metrics.
	'''
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
				"tsr": None,
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
				"terminal_enrichment": None
			}

	data = atlas.get_data()
	if failed:
		return results
					
	results["n_terminal_states"] = data.obsm["fate_probabilities"].shape[1]

	# no fate probabilities	
	if results["n_terminal_states"] <= 0:
		return results	

	#SUPERVISED - JSD
	if fixed_terminal:
		jsd = js_distance(data.obsm["fate_probabilities"], true_probabilities)
		if not jsd.index.equals(data.obs["rna:pop"].index):
			jsd = jsd.loc[data.obs["rna:pop"].index]
		jsdf = pd.DataFrame({"jsd": jsd, "cluster": data.obs["rna:pop"]})
		jsdf["potency"] = jsdf["cluster"].map(POTENCY_DICT[tree])
		results["jsd"] = jsdf.groupby("potency")["jsd"].mean()

	#SUPERVISED - TERMINAL STATE SCORE
	tts, ttp, tsr, ttc, overall = terminal_state_score(data.obs["rna:pseudotime"], data.obs["rna:pop"], ts_dict, terminal_clusters)
	results["tts"] = tts
	results["ttp"] = ttp
	results["tsr"] = tsr
	results["ttc"] = ttc
	results["temporal_state_score"] = overall

	#UNSUPERVISED
	stat, pval, _ = spearman_correlation(data.obs["rna:pseudotime"], data.obs["kl_divergence"], seed = atlas.random_state)	
	results["spearman_stat_KLD"] = stat
	results["spearman_pval_KLD"] = pval
	stat, pval, _ = spearman_correlation(data.obs["rna:pseudotime"], data.obs["shannon_entropy"], seed = atlas.random_state)	
	results["spearman_stat_SHE"] = stat
	results["spearman_pval_SHE"] = pval
	stat, pval, _ = pearson_correlation(data.obs["rna:pseudotime"], data.obs["kl_divergence"], seed = atlas.random_state)	
	results["pearson_stat_KLD"] = stat
	results["pearson_pval_KLD"] = pval
	stat, pval, _ = pearson_correlation(data.obs["rna:pseudotime"], data.obs["shannon_entropy"], seed = atlas.random_state)	
	results["pearson_stat_SHE"] = stat
	results["pearson_pval_SHE"] = pval
	stat, pval, _, __ = fate_concentration_index(data.obsm["fate_probabilities"], data.obs["rna:pseudotime"], seed = atlas.random_state)
	results["fate_index_stat"] = stat
	results["fate_index_pval"] = pval
	results["terminal_silhouette_soft"] = terminal_state_silhouette(data.obsm["fate_probabilities"], soft_assignment=True)
	results["terminal_silhouette_pse"] = terminal_state_silhouette(data.obsm["fate_probabilities"], soft_assignment=False, pseudotime=data.obs["rna:pseudotime"])
	results["terminal_enrichment"] = terminal_pseudotime_enrichment_score(terminal_states = ts_dict, pseudotime = data.obs["rna:pseudotime"],rank=True )	
	

	# VISUALIZATION
	atlas.plot_embedding(embedding_key = "X_umap",
						observation = "kl_divergence",
						save = f"_pseudokernel_KLDIV_{code}.png",
						show= False)
	atlas.plot_embedding(embedding_key = "X_umap",
						observation = "shannon_entropy",
						save = f"_pseudokernel_SHENTR_{code}.png",
						show= False)
	atlas.plot_fate_probabilities(embedding_key= "X_umap",
									states = None,
									save =  f"_pseudokernel_fates_{code}.png",
									show= False)
	try:
		atlas.plot_tree(embedding_key = "umap",
						save = f"_pseudokernel_{code}.png",
						color = "rna:pop", 
						color_milestones = False,
						show = False)
	except (IndexError, KeyError, ValueError) as e:
		print(f"WARNING: plot_tree failed for {code}: {e}")		
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
	parser.add_argument("--knn_activity", type=int)
	parser.add_argument("--wnn", type=int)
	args = parser.parse_args()
	
	# DATA CONSTRUCTION
	tree = args.tree
	diff_cif_fraction, cif_sigma = args.rd, args.sigma
	knn_rna, knn_activity, wnn = args.knn_rna, args.knn_activity, args.wnn
	n_pcs_rna, n_pcs_activity = 20, 10
	
	data_path = os.path.join(working_directory, "data", "simulated_data", tree)
	saving_simulation_path = os.path.join(working_directory, "output", "simulations", "pseudotime_kernel")
	activity = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_activity.tsv"), sep="\t", header=0, index_col=0)
	spliced = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_spliced.tsv"), sep="\t", header=0, index_col=0)
	metadata = pd.read_csv(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_metadata.tsv"), sep="\t", header=0, index_col=0)

	# create activity matrix
	activity = AnnData(X=csr_matrix(activity.values), 
			obs = pd.DataFrame(data=None, index=activity.index.values, columns=None), 
			var = pd.DataFrame(data=None, index=activity.columns, columns=None))	
	sc.pp.normalize_total(activity)
	sc.pp.pca(activity, random_state=seed, use_highly_variable=False)
	# create rna matrix
	rna = AnnData(X=csr_matrix(spliced.values.T), 
			obs=pd.DataFrame(data=None, index=spliced.columns, columns=None), 
			var=pd.DataFrame(data=None, index=spliced.index.values, columns=None))
	rna.obs = rna.obs.merge(metadata, how="left", left_index=True, right_index=True)
	sc.pp.normalize_total(rna)
	sc.pp.log1p(rna)
	sc.pp.pca(rna, random_state=seed, use_highly_variable=False)

	data = MuData({"rna":rna, "activity":activity})

	# ground truth construction
	early_cell = initial_macrostate(mudata = data, 
									pseudotime_key = "rna:pseudotime", 
									n_cells=30)
	cluster_subset = data.obs["rna:pop"].loc[early_cell].value_counts().idxmax()
	early_cell = {cluster_subset : early_cell}

	terminal_cells = {}
	for branch in TERM_DICT[tree]:
		cell = terminal_macrostate(mudata = data,
									pseudotime_key = "rna:pseudotime", 
									cluster_key = "rna:pop",
									terminal_state = branch,
									n_cells=30)
		terminal_cells[branch] = cell
	true_fates = truth_like_fates(pseudotime = data.obs["rna:pseudotime"],
									membership = data.obs["rna:pop"],
									tree = tree)

	truth_dictionary = { "fate_probabilities" :true_fates, 
						"true_states": {"initial" : early_cell,	
										"terminal": terminal_cells}
						}

	# INITIALIZATION
	start_i_wall, start_i_cpu = time.perf_counter(), time.process_time()
	tracemalloc.start()
	atlas = ATLAS(mudata = data,
					method= "pseudotime-kernel",
					fragment_path = None,
					random_state = seed,
					pseudotime_key = "rna:pseudotime",
					connectivity_key = "wnn_connectivities",
					cluster_key = "rna:pop")
	_, init_mem_peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()
	init_mem_peak = init_mem_peak / (1024 * 1024)  # bytes -> MiB
	end_i_wall, end_i_cpu = time.perf_counter(), time.process_time()
	init_wall, init_cpu = end_i_wall - start_i_wall, end_i_cpu - start_i_cpu

	# PREPROCESSING
	start_p_wall, start_p_cpu = time.perf_counter(), time.process_time()
	tracemalloc.start()
	atlas.preprocessing(n_pcs_rna = n_pcs_rna,
						n_pcs_act = n_pcs_activity,
						knn_rna = knn_rna,
						knn_act = knn_activity,
						n_neighbors = wnn) 
	_, preprocessing_mem_peak = tracemalloc.get_traced_memory()
	tracemalloc.stop()
	preprocessing_mem_peak = preprocessing_mem_peak / (1024 * 1024)  # bytes -> MiB
	end_p_wall, end_p_cpu = time.perf_counter(), time.process_time()
	preprocessing_wall, preprocessing_cpu = end_p_wall - start_p_wall, end_p_cpu - start_p_cpu

	# RUN WITH NO FIXED TERMINAL
	try:
		start_r_wall, start_r_cpu = time.perf_counter(), time.process_time()
		tracemalloc.start()
		atlas.run(connectivity_key = "wnn_connectivities",
			threshold_scheme = "hard", 
			n_states = None, 
			n_jobs = threads, 
			allow_overlap = True)
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

	ts_dict = atlas.get_data().uns.get("terminal_states", {})
	resources = {"init_wall_time": init_wall,
					"init_cpu_time": init_cpu,
					"preprocessing_wall_time": preprocessing_wall,
					"preprocessinf_cpu_time": preprocessing_cpu,
					"run_wall_time": run_wall,
					"run_cpu_time": run_cpu,
					"init_mem_peak": init_mem_peak,
					"run_mem_peak": run_mem_peak,
					"preprocessing_mem_peak": preprocessing_mem_peak
				}
	results = _apply_metrics_and_visualize(atlas = atlas, tree = tree, rd= diff_cif_fraction, sigma = cif_sigma,
					knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
					failed = failed, fixed_terminal = False, terminal_clusters = TERM_DICT[tree],
					ts_dict = ts_dict, true_probabilities = true_fates)
	_save_simulation(atlas = atlas, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
					knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = False, 
					saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary, resources = resources)


	# RUN WITH FIXED TERMINAL
	try: 
		start_r_wall, start_r_cpu = time.perf_counter(), time.process_time()
		tracemalloc.start()
		atlas.run(connectivity_key = "wnn_connectivities",
			threshold_scheme = "hard",
			n_jobs = threads,
			initial_states = early_cell, 
			terminal_states = terminal_cells)
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

	ts_dict = atlas.get_data().uns.get("terminal_states", {})
	resources = {"init_wall_time": init_wall,
					"init_cpu_time": init_cpu,
					"preprocessing_wall_time": preprocessing_wall,
					"preprocessinf_cpu_time": preprocessing_cpu,
					"run_wall_time": run_wall,
					"run_cpu_time": run_cpu,
					"init_mem_peak": init_mem_peak,
					"run_mem_peak": run_mem_peak,
					"preprocessing_mem_peak": preprocessing_mem_peak
				}
	results = _apply_metrics_and_visualize(atlas = atlas, tree = args.tree, rd= diff_cif_fraction, sigma = cif_sigma,
					knn_rna = knn_rna, knn_activity = knn_activity, wnn = wnn,
					failed = failed, fixed_terminal = True, terminal_clusters = TERM_DICT[tree],
					ts_dict = ts_dict, true_probabilities = true_fates)
	_save_simulation(atlas = atlas, tree = args.tree, rd = diff_cif_fraction, sigma= cif_sigma, 
					knn_rna = knn_rna, knn_activity = knn_activity, wnn= wnn, fixed_terminal = True, 
					saving_folder = saving_simulation_path, results=results, ground_truth = truth_dictionary, resources=resources)
