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

def check_differences(data_dict):
    # Create a set to store unique strings
    unique_strings = set()

    # Iterate through each list of strings in the dictionary
    for string_list in data_dict.values():
        # Update the set with strings from the current list
        unique_strings.update(string_list)

    # Check if the number of unique strings is greater than the total strings
    # If so, there are differences
    if len(unique_strings) > sum(len(strings) for strings in data_dict.values()):
        return True  # There are different strings
    else:
        return False  # All strings are the same


if __name__=="__main__":
	seed=42
	np.random.seed(seed)
	state = np.random.get_state()
	
	grid = {"num_waypoints": [100, 300, 500, 700, 1000], 
		"knn": [10, 30, 50, 70, 100]}	
	data_path = os.path.join(os.getcwd(), "scMultiSim")
	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu"))
	early_cell = np.random.choice(data.obs_names[data.obs["rna:pseudotime"]<0.1])
	cell53 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_3") & (data.obs["rna:pseudotime"]>0.9)])
	cell52 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_2") & (data.obs["rna:pseudotime"]>0.9)])
	cell41 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="4_1") & (data.obs["rna:pseudotime"]>0.9)])
	terminal_states = [cell53, cell52, cell41]

	failures = []
	fix_terminal = True
	pw = PalantirWrapper()
	branch = {"4-5-2": ["4_5", "5_2"], "4-5-3": ["4_5", "5_3"], "4-1": ["4_1"]}
	
	# Multiomics kernel
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data, seed=seed)
	pw.determine_multiscale_space(data)

	# RNA kernel
	pw.compute_kernel(data["rna"], knn_key ="neighbors", distance_key="distances")
	pw.run_diffusion_maps(data["rna"], seed=seed)
	pw.determine_multiscale_space(data["rna"])

	
	for values in itertools.product(*grid.values()):
		n_waypoints, knn = values
		save_multiomics = os.path.join(os.getcwd(), "results_grid", f"{n_waypoints}_{knn}_multiomics")
		save_rna = os.path.join(os.getcwd(), "results_grid", f"{n_waypoints}_{knn}_rna")	

		if not os.path.exists(save_multiomics):
			os.mkdir(save_multiomics)
		if not os.path.exists(save_rna):
			os.mkdir(save_rna)

		try:
			# Multiomics run
			if not fix_terminal:
				pw.run_palantir(data, early_cell = early_cell, num_waypoints = n_waypoints, knn=knn, seed=seed)
			else:
				pw.run_palantir(data, early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn, seed=seed)
				results = _save_results(data, saving_path= os.path.join(save_multiomics, f"results_{n_waypoints}_{knn}.tsv"), entropy_key = "palantir_entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = None, return_frame=True, group_key = "rna:pop", true_pseudotime="rna:pseudotime")

				plot_branch_correlation(dataframe = results, key1 = "rna:pseudotime", key2 = "palantir_pseudotime", group_key = "rna:pop", branch=branch, saving_path= save_multiomics, title=f"Pseudotime {n_waypoints} {knn}", xlabel = "True Pseudotime", ylabel = "Palantir Pseudotime", xlim = (0, 1.05), ylim = (0,1.05))
				ylim = (0, results["palantir_entropy"].max() + 0.05)
				plot_branch_correlation(dataframe = results, key1 = "rna:pseudotime", key2 = "palantir_entropy", group_key = "rna:pop", branch=branch, saving_path= save_multiomics, title=f"Entropy {n_waypoints} {knn}", xlabel = "True Pseudotime", ylabel = "Palantir Entropy", xlim = (0, 1.05), ylim = ylim)

			plot_palantir_results(data = data, modality_key=None, embedding_key= "X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key = "palantir_entropy", fate_prob_key = "palantir_fate_probabilities", save = True, saving_path = save_multiomics)

	
			# RNA run
			if not fix_terminal:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn)
			else:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, terminal_states=terminal_states, knn=knn)
				results = _save_results(data, saving_path= os.path.join(saving_rna, f"results_{n_waypoints}_{knn}.tsv"), entropy_key = "entropy", pseudo_time_key = "palantir_pseudotime", fate_prob_key = "palantir_fate_probabilities", modality_key = "rna", return_frame=True, group_key = "pop", true_key="pseudotime")

				plot_branch_correlation(dataframe = results, key1= "pseudotime", key2 ="palantir_pseudotime", group_key="pop", branch=brach, saving_path = saving_rna, title = f"Pseudotime {n_waypoints} {knn}", xlabel = "True Pseudotime", ylabel = "Palantir Pseudotime", xlim = (0, 1.05), ylim= (0, 1.05))
				ylim = (0, results["palantir_entropy"].max() + 0.05)
				plot_branch_correlation(dataframe = results, key1 = "pseudotime", key2 = "palantir_entropy", group_key = "pop", branch = branch, saving_path = saving_rna, title= f"Entropy {n_waypoints} {knn}", xlabel = "True Pseudotime", ylabel = "Palantir Pseudotime", xlim = (0, 1.05), ylim =ylim) 

			
			plot_palantir_results(data=data, modality_key = "rna", embedding_key="X_umap", pseudo_time_key = "palantir_pseudotime", entropy_key="palantir_entropy", fate_prob_key="palantir_fate_probabilities", save = True, saving_path = saving_rna)

		except:
			failures.append((n_waypoints, knn))

		print(failures)
