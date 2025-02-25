import os
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from palantirModel.palantir_wrapper import PalantirWrapper 
from palantirModel.palantir_plots import plot_palantir_results, correlation_plot
from palantirModel.palantir_utils import PalantirComparator

def r_squared(truth, infer):
	mean_truth = np.mean(truth)
	denom = (truth - mean_truth)**2
	num = (truth - infer)**2
	r_value = 1 - np.sum(num)/np.sum(denom)
	return r_value  

if __name__=="__main__":
	seed=42
	np.random.seed(seed)
	
	data_path = ... 
	saving_path_multiomics = os.path.join(os.getcwd(), "palantir_results_multiomics")
	saving_path_rna = os.path.join(os.getcwd(), "palantir_results_rna")

	data = mu.read_h5mu(os.path.join(data_path, "data.h5mu")) 
	
	# Palantir run
	if not os.path.exists(saving_path_multiomics):
		os.mkdir(saving_path_multiomics)
	if not os.path.exists(saving_path_rna):
		os.mkdir(saving_path_rna)

	# Select cells 
	early_cell = np.random.choice(data.obs_names[data.obs["rna:pseudotime"]<0.1])
	cell53 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_3") & (data.obs["rna:pseudotime"]>0.9)])
	cell52 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_2") & (data.obs["rna:pseudotime"]>0.9)])
	cell41 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="4_1") & (data.obs["rna:pseudotime"]>0.9)])
	terminal_states = [cell53, cell52, cell41]

	fix_terminal = True
	pw = PalantirWrapper()

	# Multiomics run
	pw.compute_kernel(data)
	pw.run_diffusion_maps(data)
	pw.determine_multiscale_space(data)
	if fix_terminal:
		pw.run_palantir(data, early_cell = early_cell, num_waypoints = 500, terminal_states = terminal_states)
	else:
		pw.run_palantir(data, early_cell = early_cell, num_waypoints=500)

	plot_palantir_results(data, saving_path = saving_path_multiomics, entropy_key=["palantir_entropy"])
	r_value = r_squared(data.obs["rna:pseudotime"].values, data.obs["palantir_pseudotime"].values)
	print("Multiomics R^2: {:.2f}".format(r_value))
	
	# RNA run
	pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key="distances")
	pw.run_diffusion_maps(data["rna"])
	pw.determine_multiscale_space(data["rna"])
	if fix_terminal:
		pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = 500, terminal_states=terminal_states)
	else:
		pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = 500)

	plot_palantir_results(data, modality_key = "rna", saving_path = saving_path_rna, entropy_key=["palantir_entropy"])
	r_value = r_squared(data.obs["rna:pseudotime"].values, data["rna"].obs["palantir_pseudotime"].values)
	print("RNA R^2: {:.2f}".format(r_value))

	if fix_terminal:
		pc = PalantirComparator()
		pc.save_palantir_matrix(data, data["rna"], "rna:pop", saving_path = os.path.join(saving_path_multiomics, "fates.tsv"))
		pc.save_palantir_matrix(data, data["rna"], "rna:pop", is_fate=False, key1 = "palantir_entropy", key2="palantir_entropy", saving_path = os.path.join(saving_path_multiomics, "entropy.tsv"))
	correlation_plot(data, pseudotime_key="rna:pseudotime", entropy_key="palantir_entropy", group_key="rna:pop", save=True, saving_path=saving_path_multiomics)
	correlation_plot(data, modality_key = "rna", pseudotime_key = "pseudotime", entropy_key="palantir_entropy", group_key="pop", save=True, saving_path = saving_path_rna)
	
