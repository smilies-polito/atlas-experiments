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


	# Correlatione entropy - pseudotime - true pseudotime
	results["no_terminal"]={} 
	results["no_terminal"]["pearson_pseudotime"] = compute_correlation(dataframe, "pearson", "pseudotime", "palantir_pseudotime")
	results["no_terminal"]["pearson_entropy"] = compute_correlation(dataframe, "pearson", "pseudotime", "palantir_entropy")
	results["no_terminal"]["kendall-tau_pseudotime"] = compute_correlation(dataframe, "kendall_tau", "pseudotime", "palantir_pseudotime")
	results["no_terminal"]["kendall-tau_entropy"] = compute_correlation(dataframe, "kendall_tau", "pseudotime", "palantir_entropy")

	results_df = dataframe[["palantir_entropy", "palantir_pseudotime", "pseudotime", "pop"]]
	results_df.columns = ["entropy", "pseudotime", "true_pseudotime", "celltype"]

	#### SET TERMINAL STATES ############################################################################
	print("SI TERMINAL")
	np.random.seed(seed)
	pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn, seed=seed)
	
	dataframe = _save_results(data, entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", group_key = "pop", true_pseudotime="pseudotime")

	saving_folder_terminal= os.path.join(saving_folder, "terminal")
	if not os.path.exists(saving_folder_terminal):
		os.mkdir(saving_folder_terminal)
	plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_folder_terminal)

	# Correlazione entropy - pseudotime - true pseudotime
	results["si_terminal"]={} 
	results["si_terminal"]["pearson_pseudotime"] = compute_correlation(dataframe, "pearson", "pseudotime", "palantir_pseudotime")
	results["si_terminal"]["pearson_entropy"] = compute_correlation(dataframe, "pearson", "pseudotime", "palantir_entropy")
	results["si_terminal"]["kendall-tau_pseudotime"] = compute_correlation(dataframe, "kendall_tau", "pseudotime", "palantir_pseudotime")
	results["si_terminal"]["kendall-tau_entropy"] = compute_correlation(dataframe, "kendall_tau", "pseudotime", "palantir_entropy")

	# f1 score
	results["f1"] = {}
	for branch, terminal in cells["terminal"].items():
		results["f1"][branch] = compute_f1(dataframe[terminal], ground_truth[branch])
	

	results_df["entropy_fixed"]=dataframe["palantir_entropy"]
	results_df["pseudotime_fixed"] = dataframe["palantir_pseudotime"]

	
	path = os.path.join(saving_folder, "results.json")
	with open(path, "w") as f:
		json.dump(results, f)
		f.close()
	
	path = os.path.join(saving_folder, "results.tsv")
	results_df.to_csv(path, sep="\t", header=True, index=True)	
