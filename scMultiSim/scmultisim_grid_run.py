import os
import gc
import itertools
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from core.palantirModel.plots import plot_palantir_results
from core.palantirModel.palantir_wrapper import PalantirWrapper 
from core.palantirModel.utils import _save_results
from core.utils import plot_branch_correlation


if __name__=="__main__":
	seed=42
	np.random.seed(seed)
	state = np.random.get_state()
	
	grid = {"diff_cif_fraction": [.1,.3,.5,.7,.9,], 
		"cif_sigma": [.1,.3,.5,.7,.9]}	
	data_path = os.path.join(os.getcwd(), "scMultiSim", "phyla5")
	early_cell = "cell613"
	cell53 = "cell594"
	cell52 = "cell381"
	cell41 = "cell698" 
	terminal_states = [cell53, cell52, cell41]

	failures = []
	n_waypoints, knn = 500, 30
	fix_terminal = True
	pw = PalantirWrapper()
	branch = {"4-5-2": ["4_5", "5_2"], "4-5-3": ["4_5", "5_3"], "4-1": ["4_1"]}

	for values in itertools.product(*grid.values()):
		diff_cif_fraction, cif_sigma = values
		data = mu.read_h5mu(os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_data.h5mu"))
		# Multiomics kernel
		pw.compute_kernel(data)
		pw.run_diffusion_maps(data, seed=seed)
		pw.determine_multiscale_space(data)

		# RNA kernel
		pw.compute_kernel(data["rna"], knn_key ="neighbors", distance_key="distances")
		pw.run_diffusion_maps(data["rna"], seed=seed)
		pw.determine_multiscale_space(data["rna"])

		save_multiomics = os.path.join(os.getcwd(), "results_grid", f"{diff_cif_fraction}_{cif_sigma}_multiomics")
		save_rna = os.path.join(os.getcwd(), "results_grid", f"{diff_cif_fraction}_{cif_sigma}_rna")	

		if not os.path.exists(save_multiomics):
			os.mkdir(save_multiomics)
		if not os.path.exists(save_rna):
			os.mkdir(save_rna)
		

		try:
			# Multiomics run
			if not fix_terminal:
				pw.run_palantir(data, early_cell = early_cell, num_waypoints = n_waypoints, knn=knn, seed=seed)
				results = _save_results(data, saving_path= os.path.join(save_multiomics, f"results_{diff_cif_fraction}_{cif_sigma}.tsv"), entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = None, return_frame=True, group_key = "rna:pop", true_pseudotime="rna:pseudotime")
			else:
				pw.run_palantir(data, early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn, seed=seed)
				results = _save_results(data, saving_path= os.path.join(save_multiomics, f"results_{diff_cif_fraction}_{cif_sigma}.tsv"), entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = None, return_frame=True, group_key = "rna:pop", true_pseudotime="rna:pseudotime")

				plot_branch_correlation(dataframe = results, key1 = "rna:pseudotime", key2 = "palantir_pseudotime", group_key = "rna:pop", branch=branch, saving_path= save_multiomics, title=f"Pseudotime sigma={cif_sigma} rd={diff_cif_fraction}", xlabel = "True Pseudotime", ylabel = "Palantir Pseudotime", xlim = (0, 1.05), ylim = (0,1.05))
				ylim = (0, results["palantir_entropy"].max() + 0.05)
				plot_branch_correlation(dataframe = results, key1 = "rna:pseudotime", key2 = "palantir_entropy", group_key = "rna:pop", branch=branch, saving_path= save_multiomics, title=f"Entropy sigma={cif_sigma} rd={diff_cif_fraction}", xlabel = "True Pseudotime", ylabel = "Palantir Entropy", xlim = (0, 1.05), ylim = ylim)

			plot_palantir_results(data = data, modality_key=None, embedding_key= "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save = True, saving_path = save_multiomics)

	
			# RNA run
			if not fix_terminal:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn)
				results = _save_results(data, saving_path= os.path.join(save_rna, f"results_{diff_cif_fraction}_{cif_sigma}.tsv"), entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", return_frame=True, group_key = "pop", true_pseudotime="pseudotime")
			else:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, terminal_states=terminal_states, knn=knn)
				results = _save_results(data, saving_path= os.path.join(save_rna, f"results_{diff_cif_fraction}_{cif_sigma}.tsv"), entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", return_frame=True, group_key = "pop", true_pseudotime="pseudotime")
				plot_branch_correlation(dataframe = results, key1= "pseudotime", key2 ="palantir_pseudotime", group_key="pop", branch=branch, saving_path = save_rna, title = f"Pseudotime sigma={cif_sigma} rd={diff_cif_fraction}", xlabel = "True Pseudotime", ylabel = "Palantir Pseudotime", xlim = (0, 1.05), ylim= (0, 1.05))
				ylim = (0, results["palantir_entropy"].max() + 0.05)
				plot_branch_correlation(dataframe = results, key1 = "pseudotime", key2 = "palantir_entropy", group_key = "pop", branch = branch, saving_path = save_rna, title= f"Entropy sigma={cif_sigma} rd={diff_cif_fraction}", xlabel = "True Pseudotime", ylabel = "Palantir Pseudotime", xlim = (0, 1.05), ylim =ylim) 

			
			plot_palantir_results(data=data, modality_key = "rna", embedding_key="X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key="palantir_entropy", fate_prob_key="palantir_fate_probabilities", save = True, saving_path = save_rna)

		except Exception as e:
			failures.append((diff_cif_fraction, cif_sigma))
			print(e)

	print(failures)
