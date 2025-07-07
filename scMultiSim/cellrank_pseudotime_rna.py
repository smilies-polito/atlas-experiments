import os
import json
import time
import cellrank
import muon as mu
import scanpy as sc
import pandas as pd
import argparse 
import numpy as np
from itertools import product
from core.cellrankPseudotime.pseudokernel import PseudotimeKernelMuon
from core.matrix_analysis import MatrixAnalyser
from core.palantirModel.plots import plot_palantir_results
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.metrics import compute_correlation, compute_f1

if __name__=="__main__":
	seed = 42	
	np.random.seed(seed)

	grid_preprocessing = {"diff_cif_fraction":[0.1, 0.3, 0.5, 0.7, 0.9], "sigma_cif":[0.1, 0.3, 0.5, 0.7, 0.9], "knn_rna":[30, 50, 70, 100], "knn_atac":[30, 50, 70, 100], "wnn":[30, 50, 70, 100]}
	grid_palantir = {"percentage":[0.25, 0.5, 0.75], "knn_waypoints":[30,50,100]}

	parser = argparse.ArgumentParser()
	parser.add_argument("--n_jobs", type=int, help ="Slurm cpus-per-task")
	args = parser.parse_args()

	#arguments for results construction
	fix_terminal = ...
	threshold_scheme = ..
	n_macrostates = ...
	albero = ...
	palantir_early_cell = ...
	results = {"code": None, "albero":albero, "n_cellule":1000, "GRN_type":"GRN_100", "sigma_cif":None, "diff_cif_fraction":None, "modello":"rna", "fixed_terminal":fix_terminal, "knn_atac":None, "knn_rna":None, "wnn":None, "algoritmo":"Cellrank Pseudotime Kernel", "n_waypoints":None, "knn_waypoints":None, "n_macrostates":n_macrostates, "velocity_algorithm":None, "pruning_type":threshold_scheme, "pearson_kl_statistics":None, "pearson_kl_pvalue":None, "pearson_entropy_statistics":None, "pearson_entropy_pvalue":None, "f1_cosine":None, "f1_euclidean":None, "cpu_time": None, "wall_time":None, "cpu_time_kernel":None, "wall_time_kernel":None}
	
	#path declaration + additional files
	working_dir = ...
	failures_path = os.path.join(working_dir, "results", "failing_pseudoKernel_simulations.csv")
	tsv_path = os.path.join(working_dir, "results", "pseudoKernel_simulations.csv") 
 	data_path = os.path.join(working_dir, "data", f"{albero}")

#	saving_folder = os.path.join( ... )
#	if not os.path.exists(saving_folder):
#		os.mkdir(saving_folder)

	# read selected cells 
	cell_path = os.path.join(data_path, "selected_cells.json")
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	
	# read ground truth fates probabilities 
	truth_path = os.path.join(data_path, "branch_assignment.tsv")
	ground_truth = pd.read_csv(truth_path, sep= "\t", index_col=0, header=0)

	# read dataset
	for diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn in product(*grid_preprocessing.values()):
		results["sigma_cif"] = sigma_cif
		results["diff_cif_fraction"] = diff_cif_fraction
		results["knn_atac"] = knn_atac
		results["knn_rna"] = knn_rna
		results["wnn"] = wnn
		data = mu.read_h5mu(os.path.join(data_path, f"{knn_rna}_{knn_atac}_{wnn}", "{diff_cif_fraction}_{sigma_cif}.h5mu"))
 		data["rna"].obsm["X_umap"] = data.obsm["X_umap"] 
		for percentage, knn_waypoints in product(*grid_palantir.values()):
			n_waypoints = int(data.shape[0] * percentage)
			results["n_waypoints"]=n_waypoints
			results["knn_waypoints"] = knn_waypoints
			start_wall_kernel, end_wall_kernel  = None, None
			start_cpu_kernel , end_cpu_kernel = None, None
			start_wall_gpcca, end_wall_gpcca = None, None
			start_cpu_gpcca, end_cpu_gpcca= None, None

            #execute diffusion maps: not considered in time computation paalantir execution
			pw = PalantirWrapper()
			pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key="distances")
			pw.run_diffusion_maps(data["rna"], seed=seed)
			pw.determine_multiscale_space(data["rna"])
	
			try:
				pw.run_palantir(data["rna"], early_cell = palantir_early_cell, num_waypoints = n_waypoints, knn=knn_waypoints, seed=seed)	
				data.update()

				start_wall_kernel = time.time() 
				start_cpu_kernel = time.process_time()
		
				kernel = PseudotimeKernelMuon(data=data, modality_key="rna", embedding_key="X_umap", connectivity_key = "connectivities", pseudotime_key = "palantir_pseudotime", group_key=["pop"])
				kernel.compute_transition_matrix(threshold_scheme=threshold_scheme)
			
				end_wall_kernel = time.time()
				end_cpu_kernel = time.process_time()
				results["wall_time_kernel"] = end_wall_kernel - start_wall_kernel
				results["cpu_time_kernel"] = end_cpu_kernel - start_cpu_kernel

				start_wall_gpcca = time.time()
				start_cpu_gpcca = time.process_time()
			
				g = cellrank.estimators.GPCCA(kernel.kernel)
				g.compute_schur()
				if fix_terminal:
					g.set_initial_states(cells["initial"])
					g.set_terminal_states(cells["terminal"])
				else:
					g.compute_macrostates(n_states = n_macrostates, cluster_key="pop")
					g.predict_terminal_states()
					g.predict_initial_states(allow_overlap=True)
				g.compute_fate_probabilities(tol = 1e-10, preconditioner="ilu", n_jobs=args.n_jobs, use_petsc=True, show_progress_bar=False, backend="threading")

				end_wall_gpcca = time.time()
				end_cpu_gpcca = time.process_time()
		
				results["wall_time"] = (end_wall_gpcca - start_wall_gpcca) + (end_wall_kernel  - start_wall_kernel)
				results["cpu_time"] = (end_cpu_gpcca - start_cpu_gpcca) + (end_cpu_kernel - start_cpu_kernel )

				dataframe = pd.DataFrame(g.fate_probabilities.X, columns = g.fate_probabilities.names, index=data.obs_names)
	
				# if fix terminal then compute f1 score	
				if fix_terminal:
					keys = list(cells["terminal"].keys())
					cosine, euclidean = compute_f1(dataframe[keys], ground_truth[keys], aggregate=True)
					results["f1_cosine"] = cosine
					results["f1_euclidean"] = euclidean
				else: 
					results["f1_cosine"] = None
					results["f1_euclidean"] = None

				dataframe["pseudotime"] = data.obs["rna:palantir_pseudotime"]
				dataframe["celltype"] = data.obs["rna:pop"]
				dataframe["entropy"] = g.compute_lineage_priming(method="entropy")
				dataframe["KL"] = g.compute_lineage_priming(method="kl_divergence")
				dataframe = dataframe[["entropy", "pseudotime", "KL", "celltype"]]

#		 		plot images
#				saving_path= os.path.join( ... )
#				if not os.path.exists(saving_path):
#					os.mkdir(saving_path)
#               g.plot_fate_probabilities(same_plot=True, save=os.path.join( ... ), title="Fate Probabilities", show=False)
#               if not fix_terminal:
#                   g.plot_macrostate_composition(key="pop", show=False, save=os.path.join( ... ), title=f"Macrostate Composition {n_macrostates}")
#                   g.plot_coarse_T(annotate=True, save = os.path.join( ... ), title=f"Macrostates {n_macrostates}")

				# Compute correlations and save results
				statistics, pvalue= compute_correlation(dataframe, "pearson", "pseudotime", "KL")
				results["pearson_kl_statistics"] = statistics
				results["pearson_kl_pvalue"] = pvalue

				statistics, pvalue = compute_correlation(dataframe, "pearson", "pseudotime", "entropy")
				results["pearson_entropy_statistics"] = statistics 
				results["pearson_entropy_pvalue"] = pvalue
	
				results["code"] = results.get("albero") + "_" + str(results.get("diff_cif_fraction")) + ":" + str(results.get("sigma_cif")) + "_" + str(results.get("knn_rna")) + ":" + str(results.get("knn_atac")) + ":" + str(results.get("wnn")) + "_" + str(results.get("n_waypoints")) + ":"+ str(results.get("knn_waypoints")) + "_" + str(results.get("fixed_terminal"))

				pd.DataFrame(results, index=[0]).to_csv(tsv_path, sep=",", header=False, index=False, mode="a")	
			
			except: 
				pd.DataFrame([(diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn_waypoints, fix_terminal)], index=[0]).to_csv(failures_path, sep=",", header=False, index=False, mode="a")

