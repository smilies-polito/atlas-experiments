import os
import gc
import json
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from core.palantirModel.plots import plot_palantir_results
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.utils import _save_results
from core.metrics import compute_correlation, compute_f1
from core.utils import save_run_results

if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	results = {}	
	diff_cif_fraction, sigma_cif = ...
	n_waypoints, knn = ... 
	
	data_path =  ... 
	saving_folder = ... 
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

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
	
	# read data + kernel + diffusion space
	data = mu.read_h5mu(...)
	pw.compute_kernel(data["rna"], distance_key="connectivities", knn_key="neighbors")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	#### NO TERMINAL STATES #############################################################################
	print("TERMINAL NO")
	np.random.seed(seed)
	pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn, seed=seed)
	
	dataframe = _save_results(data, entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", group_key = "pop", true_pseudotime="pseudotime")

	saving_folder_noterminal= os.path.join(saving_folder, "no_terminal")
	if not os.path.exists(saving_folder_noterminal):
		os.mkdir(saving_folder_noterminal)
	plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_folder_noterminal)

	dataframe = dataframe[["palantir_entropy", "palantir_pseudotime", "pseudotime", "pop"]]
	dataframe.columns = ["entropy", "pseudotime", "true_pseudotime", "celltype"]

	# Correlation entropy - pseudotime - true pseudotime
	results["pearson_pseudotime"] = compute_correlation(dataframe, "pearson", "true_pseudotime", "pseudotime")
	results["pearson_entropy"] = compute_correlation(dataframe, "pearson", "true_pseudotime", "entropy")
	results["kendall-tau_pseudotime"] = compute_correlation(dataframe, "kendall_tau", "true_pseudotime", "pseudotime")
	results["kendall-tau_entropy"] = compute_correlation(dataframe, "kendall_tau", "true_pseudotime", "entropy")

	save_run_results(dictionary=results, dataframe=dataframe, saving_folder = saving_folder, terminal=False)


	#### SET TERMINAL STATES ############################################################################
	print("SI TERMINAL")
	np.random.seed(seed)
	pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn, seed=seed)
	
	dataframe = _save_results(data, entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", group_key = "pop", true_pseudotime="pseudotime")

	# f1 score
	results["f1"] = {}
	for branch, terminal in cells["terminal"].items():
		results["f1"][branch] = compute_f1(dataframe[terminal].values.T.reshape(1,-1), ground_truth[branch].values.T.reshape(1,-1))
	results["f1"]["total"] = compute_f1(dataframe[list(cells["terminal"].values())], ground_truth[list(cells["terminal"].keys())], aggregate=True)

	#plots
	saving_folder_terminal= os.path.join(saving_folder, "terminal")
	if not os.path.exists(saving_folder_terminal):
		os.mkdir(saving_folder_terminal)
	plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_folder_terminal)

	dataframe = dataframe[["palantir_entropy", "palantir_pseudotime", "pseudotime", "pop"]]
	dataframe.columns = ["entropy", "pseudotime", "true_pseudotime", "celltype"]

	# Correlation entropy - pseudotime - true pseudotime
	results["pearson_pseudotime"] = compute_correlation(dataframe, "pearson", "true_pseudotime", "pseudotime")
	results["pearson_entropy"] = compute_correlation(dataframe, "pearson", "true_pseudotime", "entropy")
	results["kendall-tau_pseudotime"] = compute_correlation(dataframe, "kendall_tau", "true_pseudotime", "pseudotime")
	results["kendall-tau_entropy"] = compute_correlation(dataframe, "kendall_tau", "true_pseudotime", "entropy")

	save_run_results(dictionary=results, dataframe=dataframe, saving_folder=saving_folder, terminal=True)
