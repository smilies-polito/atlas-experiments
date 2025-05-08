import os
import json
import time
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from itertools import product
from core.palantirModel.plots import plot_palantir_results
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.utils import _save_results
from core.metrics import compute_correlation, compute_f1
from core.utils import save_run_results

if __name__=="__main__":
	seed = 42	
	np.random.seed(seed)

	#arguments from command line TO DO 
	grid_preprocessing = {"diff_cif_fraction":[0.1, 0.3, 0.5, 0.7, 0.9], "sigma_cif":[0.1, 0.3, 0.5, 0.7, 0.9], "knn_rna":[30, 50, 70, 100], "knn_atac":[30, 50, 70, 100], "wnn":[30, 50, 70, 100]}
	grid_palantir = {"percentage":[0.25, 0.5, 0.75], "knn_waypoints":[30,50,100]}
	
	#arguments for results construction
	fix_terminal = ...
	albero = ... 
	results = {"albero":albero, "n_cellule":1000, "GRN_type":"GRN_100", "sigma_cif":None, "diff_cif_fraction":None, "modello":"rna", "fixed_terminal":fix_terminal, "knn_atac":None, "knn_rna":None, "wnn":None, "algoritmo":"palantir", "n_waypoints":None, "knn_waypoints":None, "n_macrostates":None, "velocity_algorithm":None, "pruning_type":None, "pearson_pseudotime_statistics":None, "pearson_pseudotime_pvalue":None, "kendall_pseudotime_statistics":None, "kendall_pseudotime_pvalue":None, "pearson_entropy_statistics":None, "pearson_entropy_pvalue":None, "f1_cosine":None, "f1_euclidean":None, "cpu_time": None, "wall_time":None}
	
	#path declaration + additional files
	tsv_path = ...
 	data_path = ... 
	failures_path = ...

#	saving_folder = ... 
#	if not os.path.exists(saving_folder):
#		os.mkdir(saving_folder)

	# read selected cells 
	cell_path = ...
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	terminal_states = list(cells["terminal"].values())
	early_cell = list(cells["initial"].values())[0]
	
	# read ground truth fates probabilities 
	truth_path = ... 
	ground_truth = ...

	# read dataset
	for diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn in product(*grid_preprocessing.values()):
		results["sigma_cif"] = sigma_cif
		results["diff_cif_fraction"] = diff_cif_fraction
		results["knn_atac"] = knn_atac
		results["knn_rna"] = knn_rna
		results["wnn"] = wnn
		data = mu.read_h5mu(...)
		data["rna"].obsm["X_umap"] = data.obsm["X_umap"] 
		for percentage, knn_waypoints in product(*grid_palantir.values()):
            # aggiungere se già letto 
			n_waypoints = int(data.shape[0] * percentage)
			results["n_waypoints"]=n_waypoints
			results["knn_waypoints"] = knn_waypoints
			start_wall_diffusion, end_wall_diffusion = None, None
			start_cpu_diffusion, end_cpu_diffusion = None, None
			start_wall_palantir, end_wall_palantir = None, None
			start_cpu_palantir, end_cpu_palantir= None, None

			# start times for diffusion process 
			start_wall_diffusion = time.time()
			start_cpu_diffusion = time.process_time()

			#execute diffusion maps
			pw = PalantirWrapper()
			pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key="distances")
			pw.run_diffusion_maps(data["rna"], seed=seed)
			pw.determine_multiscale_space(data["rna"])
	
			# stop time for diffusion process
			end_wall_diffusion = time.time()
			end_cpu_diffusion = time.process_time()

			try:
				# start time for pseudotime and fate probabilities
				start_wall_palantir = time.time()
				start_cpu_palantir = time.process_time()		 
				if fix_terminal:	
					pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn_waypoints, seed=seed)
				else:
					pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn_waypoints, seed=seed)	
		
				# stop time for pseudotime and fates probabilities + update times
				end_wall_palantir = time.time()
				end_cpu_palantir = time.process_time()
		
				results["wall_time"] = (end_wall_palantir - start_wall_palantir) + (end_wall_diffusion - start_wall_diffusion)
				results["cpu_time"] = (end_cpu_palantir - start_cpu_palantir) + (end_cpu_diffusion - start_cpu_diffusion)

				dataframe = _save_results(data, entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", group_key = "pop", true_pseudotime="pseudotime")
	
				# if fix terminal then compute f1 score	
				if fix_terminal:		
					cosine, euclidean = compute_f1(dataframe[list(cells["terminal"].values())], ground_truth[list(cells["terminal"].keys())], aggregate=True)
					results["f1_cosine"] = cosine
					results["f1_euclidean"] = euclidean

#		 		plot images
#				saving_path= ... 
#				if not os.path.exists(saving_path):
#					os.mkdir(saving_path)
#				plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_path)

				dataframe = dataframe[["palantir_entropy", "palantir_pseudotime", "pseudotime", "pop"]]
				dataframe.columns = ["entropy", "pseudotime", "true_pseudotime", "celltype"]

				# Compute correlations and save results
				statistics, pvalue= compute_correlation(dataframe, "pearson", "true_pseudotime", "pseudotime")
				results["pearson_pseudotime_statistics"] = statistics
				results["pearson_pseudotime_pvalue"] = pvalue

				statistics, pvalue = compute_correlation(dataframe, "pearson", "true_pseudotime", "entropy")
				results["pearson_entropy_statistics"] = statistics 
				results["pearson_entropy_pvalue"] = pvalue
	
				statistics, pvalue = compute_correlation(dataframe, "kendall_tau", "true_pseudotime", "pseudotime")
				results["kendall_pseudotime_statistics"] = statistics 
				results["kendall_pseudotime_pvalue"] = pvalue
		
				pd.DataFrame(results, index=[0]).to_csv(tsv_path, sep=",", header=False, index=False, mode="a")	
			
			except:
					pd.DataFrame([(diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn_waypoints, fix_terminal)], index=[0]).to_csv(failures_path, sep=",", header=False, index=False, mode="a")

