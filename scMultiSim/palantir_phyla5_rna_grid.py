import os
import gc
import json
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

	tsv_path = ... 
	grid_scmultisim = {"diff_cif_fraction":[.1,.3,.5,.7,.9] , "sigma_cif" : [.1,.3,.5,.7,.9], "knn_rna":[30, 50, 70, 100], "knn_atac":[30,50,70,100], "wnn":[30,50,70,100]}
	grid_palantir = {"n_waypoints":[.25, .5, .75], "knn": [30, 50, 100]}
	
	data_path =  ... 
	failures = []

	results = {"albero":"phyla5", "n_cellule":1000, "GRN_type":"GRN_100", "sigma_cif":None, "diff_cif_fraction":None, "modello":"rna", "fixed_terminal":False, "knn_atac":None, "knn_rna":None, "wnn":None, "algoritmo":"palantir", "n_waypoints":None, "knn_waypoints":None, "n_macrostates":None, "velocity_algorithm":None, "pruning_type":None, "pearson_pseudotime_statistics":None, "pearson_pseudotime_pvalue":None, "kendall_pseudotime_statistics":None, "kendall_pseudotime_pvalue":None, "pearson_entropy_statistics":None, "pearson_entropy_pvalue":None, "f1_cosine":None, "f1_euclidean":None}

	# read selected cells 
	cell_path = ... 
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	terminal_states = list(cells["terminal"].values())
	early_cell = list(cells["initial"].values())[0]
	
	# read ground truth fates probabilities 
	truth_path = ... 
	ground_truth = pd.read_csv(truth_path, sep= "\t", index_col=0, header=0)

	pw = PalantirWrapper()
	
	for diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn in product(*grid_scmultisim.values()):	
		results["diff_cif_fraction"] = diff_cif_fraction
		results["sigma_cif"] = sigma_cif
		results["knn_rna"] = knn_rna
		results["knn_atac"] = knn_atac
		results["wnn"] = wnn 

		# read data and determine multiscale space
		data = mu.read_h5mu(os.path.join(data_path, f"{knn_rna}_{knn_atac}_{wnn}", f"{diff_cif_fraction}_{sigma_cif}_data.h5mu"))
		data["rna"].obsm["X_umap"] = data.obsm["X_umap"] #adjust to obtain comparable plots umap
		pw.compute_kernel(data["rna"], knn_key = "neighbors", distance_key = "distances")
		pw.run_diffusion_maps(data["rna"], seed=seed)
		pw.determine_multiscale_space(data["rna"])
	
		# create results folder
		saving_folder = os.path.join(os.getcwd(), "palantir_phyla5_rna", f"{diff_cif_fraction}_{sigma_cif}_{knn_rna}{knn_atac}{wnn}_rna")
		if not os.path.exists(saving_folder):
			os.mkdir(saving_folder)
		
		# iterate over palantir parameters
		for n_waypoints, knn in product(*grid_palantir.values()):
			n_waypoints = int(data.shape[0] * n_waypoints)
			results["n_waypoints"] = n_waypoints
			results["knn_waypoints"] = knn

		#### NO TERMINAL STATES #############################################################################
			np.random.seed(seed)
			results["fixed_terminal"] = False
			try:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn, seed=seed)	
				dataframe = _save_results(data["rna"], entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = None, group_key = "pop", true_pseudotime="pseudotime")
				
				results["f1_cosine"] = None
				results["f1_euclidean"] = None

				saving_folder_noterminal= os.path.join(saving_folder, f"no_terminal_{n_waypoints}_{knn}")
				if not os.path.exists(saving_folder_noterminal):
					os.mkdir(saving_folder_noterminal)
				plot_palantir_results(data = data["rna"], modality_key = None, embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_folder_noterminal)

				dataframe = dataframe[["palantir_entropy", "palantir_pseudotime", "pseudotime", "pop"]]
				dataframe.columns = ["entropy", "pseudotime", "true_pseudotime", "celltype"]

				# Correlation entropy - pseudotime - true pseudotime
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
				failures.append((diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn, "noterminal"))

		#### SET TERMINAL STATES ############################################################################
			np.random.seed(seed)
			results["fixed_terminal"] = True
			try:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn, seed=seed)
				dataframe = _save_results(data["rna"], entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = None, group_key = "pop", true_pseudotime="pseudotime")
	
				cosine, euclidean = compute_f1(dataframe[list(cells["terminal"].values())], ground_truth[list(cells["terminal"].keys())], aggregate=True)
				results["f1_cosine"] = cosine
				results["f1_euclidean"] = euclidean
		
				# plots
				saving_folder_terminal= os.path.join(saving_folder, f"terminal_{n_waypoints}_{knn}")
				if not os.path.exists(saving_folder_terminal):
					os.mkdir(saving_folder_terminal)
				plot_palantir_results(data = data["rna"], modality_key = None, embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_folder_terminal)

				dataframe = dataframe[["palantir_entropy", "palantir_pseudotime", "pseudotime", "pop"]]
				dataframe.columns = ["entropy", "pseudotime", "true_pseudotime", "celltype"]

				# Correlazione entropy - pseudotime - true pseudotime
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
				failures.append((diff_cif_fraction, sigma_cif, knn_rna, knn_atac, wnn, n_waypoints, knn, "terminal"))

	failure_path = ... 
	pd.DataFrame(failures, columns = ["rd", "sigma", "knn_rna", "knn_atac", "wnn", "n_waypoints", "knn_waypoints", "run_type"]).to_csv(failure_path, sep="\t", index=False, header=True)
