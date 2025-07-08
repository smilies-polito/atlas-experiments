import os
import json
import time
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from itertools import product
from utils import aggregate_lineage_fate
from core.palantirModel.plots import plot_palantir_results, plot_similarity_matrix, plot_diffusion_space
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.utils import _save_results
from core.metrics import compute_correlation, compute_f1
from core.utils import save_run_results


if __name__=="__main__":
	seed = 42
	np.random.seed(seed)

	working_dir = ... 
	donor= ... #either donor1 or donor2
	data_path = os.path.join(working_dir, "data", "lineage_tracing", f"{donor}")
	results_folder = os.path.join(working_dir, "results", "lineage_tracing")

	fixed_terminal = ...
	model = "rna"
	
	#path declaration + additional files
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	data["rna"].obsm["X_umap"] = data.obsm["X_umap"]
	data["rna"].obs = data.obs[["lineage", "STD.CellType"]]

	results = {"donor": f"{donor}", "algorithm": "palantir", "model":f"{model}",
		      "fixed_terminal": fixed_terminal, "f1_cosine": None, "f1_euclidean": None,
		"pearson_entropy_statistics": None, "pearson_entropy_pvalue": None, "n_terminal_states": None}

	saving_folder = os.path.join(results_folder, f"palantir_{donor}_{model}_{fixed_terminal}")
	results_path = os.path.join(results_folder, f"palantir_results.csv")
	if not os.path.exists(saving_folder):
		os.mkdir(saving_folder)

	# select initial cell 
	cell_path = os.path.join(data_path, "selected_cells_palantir.json")
	with open(cell_path, "r") as f:
		cells = json.load(f)
		f.close()
	terminal_states = list(cells["terminal"].values())
	early_cell = list(cells["initial"].values())[0]
	
	# read ground truth fates probabilities 
	truth_path = os.path.join(data_path, "branch_assignment.csv")
	ground_truth = pd.read_csv(truth_path, sep= ",", index_col=0, header=0)

	#execute diffusion maps
	pw = PalantirWrapper()
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key = "distances")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	try:
		if fixed_terminal:
			pw.run_palantir(data["rna"], early_cell = early_cell, terminal_states = terminal_states, seed=seed)
		else:
			pw.run_palantir(data["rna"], early_cell=early_cell, seed = seed) 

		dataframe = _save_results(data, entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", group_key = "lineage", true_pseudotime=None)
		dataframe = dataframe.merge(data.obs[["STD.CellType"]], how="left", left_index=True, right_index=True)

		if fixed_terminal:
			dataframe_filtered = dataframe[dataframe.index.isin(ground_truth.index.to_list())]
			aggregate_lineage_fate(dataframe_filtered, cells["terminal"])
			lineages = ["megakaryocyte", "erythroid", "myeloid", "lymphoid"]
			cosine, euclidean = compute_f1(dataframe_filtered[lineages], ground_truth[lineages], aggregate=True)
			results["f1_cosine"] = cosine
			results["f1_euclidean"] = euclidean
		else:
			results["n_terminal_states"] = data["rna"].obsm["palantir_fate_probabilities"].shape[1]

		plot_palantir_results(data = data, modality_key = "rna", embedding_key = "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save= True, saving_path = saving_folder)

		dataframe = dataframe[["palantir_entropy", "palantir_pseudotime", "lineage"]]

		# Compute correlations and save results
		statistics, pvalue = compute_correlation(dataframe, "pearson", "palantir_pseudotime", "palantir_entropy")
		results["pearson_entropy_statistics"] = statistics 
		results["pearson_entropy_pvalue"] = pvalue
	
		pd.DataFrame(results, index=[0]).to_csv(results_path, sep=",", header=False, index=False, mode="a")	
	except Exception as e:
		print(e)

