import os
import gc
import itertools
import muon as mu
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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
	
	grid = {"num_waypoints": [100, 300, 500, 700, 1000], 
		"knn": [10, 30, 50, 70, 100]}	
	data_path = ...
	data = mu.read_h5mu( ... )
	early_cell = np.random.choice(data.obs_names[data.obs["rna:pseudotime"]<0.1])
	cell53 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_3") & (data.obs["rna:pseudotime"]>0.9)])
	cell52 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="5_2") & (data.obs["rna:pseudotime"]>0.9)])
	cell41 = np.random.choice(data.obs_names[(data.obs["rna:pop"]=="4_1") & (data.obs["rna:pseudotime"]>0.9)])
	terminal_states = [cell53, cell52, cell41]

	fix_terminal = True
	pw = PalantirWrapper()
	rvalues_pseudotime = {}
	rvalues_entropy= {}
	failing_experiments = []
	for values in itertools.product(*grid.values()):
		try:
			n_waypoints, knn = values
			saving_multiomics = os.path.join(os.getcwd(), "results_grid", f"{n_waypoints}_{knn}_multiomics")
			saving_rna = os.path.join(os.getcwd(), "results_grid", f"{n_waypoints}_{knn}_rna")	
	
			if not os.path.exists(saving_multiomics):
				os.mkdir(saving_multiomics)
			if not os.path.exists(saving_rna):
				os.mkdir(saving_rna)

			# Multiomics run
			pw.compute_kernel(data)
			pw.run_diffusion_maps(data)
			pw.determine_multiscale_space(data)
			if not fix_terminal:
				pw.run_palantir(data, early_cell = early_cell, num_waypoints = n_waypoints, knn=knn)
			else:
				pw.run_palantir(data, early_cell = early_cell, num_waypoints=n_waypoints, terminal_states = terminal_states, knn=knn)
				correlation_plot(data, modality_key=None, key1= "rna:pseudotime", key2 = "palantir_pseudotime", group_key=None, 
						save = True, saving_path = saving_multiomics)
				r_value = r_squared(data.obs["rna:pseudotime"].values, data.obs["palantir_pseudotime"].values)
				corr = pearsonr(data.obs["rna:pseudotime"].values, data.obs["palantir_pseudotime"].values)
				rvalues_pseudotime[f"multiomics_{n_waypoints}_{knn}"] = (r_value, corr.statistic, corr.pvalue)
			corr = pearsonr(data.obs["rna:pseudotime"].values, data.obs["palantir_entropy"].values)
			rvalues_entropy[f"multiomics_{n_waypoints}_{knn}"] = (corr.statistic, corr.pvalue)
			
			plot_palantir_results(data, saving_path = saving_multiomics, entropy_key=["palantir_entropy"])
		

		# RNA run
			pw.compute_kernel(data["rna"], knn_key="neighbors", distance_key="distances")
			pw.run_diffusion_maps(data["rna"])
			pw.determine_multiscale_space(data["rna"])
			if not fix_terminal:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, knn=knn)
			else:
				pw.run_palantir(data["rna"], early_cell = early_cell, num_waypoints = n_waypoints, terminal_states=terminal_states, knn=knn)
				correlation_plot(data, modality_key="rna", key1 = "pseudotime", key2="palantir_pseudotime", group_key = None, 
								save = True, saving_path = saving_rna)
				r_value = r_squared(data["rna"].obs["pseudotime"].values, data["rna"].obs["palantir_pseudotime"].values) 
				corr = pearsonr(data["rna"].obs["pseudotime"].values, data["rna"].obs["palantir_pseudotime"].values)
				rvalues_pseudotime[f"rna_{n_waypoints}_{knn}"] = (r_value, corr.statistic, corr.pvalue)
			corr = pearsonr(data["rna"].obs["pseudotime"].values, data["rna"].obs["palantir_entropy"].values)
			rvalues_entropy[f"rna_{n_waypoints}_{knn}"]= (corr.statistic, corr.pvalue)

			plot_palantir_results(data, modality_key = "rna", saving_path = saving_rna, entropy_key=["palantir_entropy"])

			if fix_terminal:
				pc = PalantirComparator()
				pc.save_palantir_matrix(data, data["rna"], "rna:pop", saving_path = os.path.join(saving_multiomics, "fates.tsv"))
				pc.save_palantir_matrix(data, data["rna"], "rna:pop", is_fate=False, key1 = "palantir_entropy", key2="palantir_entropy", saving_path = os.path.join(saving_multiomics, "entropy.tsv"))
				correlation_plot(data, key1="rna:pseudotime", key2="palantir_entropy", group_key="rna:pop", save=True, saving_path=saving_multiomics)
				correlation_plot(data, modality_key = "rna", key1 = "pseudotime", key2="palantir_entropy", group_key="pop", save=True, saving_path = saving_rna)
		except Exception as e:
			failing_experiments.append((n_waypoints, knn))
	
		gc.collect()

	pd.DataFrame(rvalues_pseudotime).T.to_csv(os.path.join(data_path, "correlations_pseudotime.tsv"), sep="\t", header=True, index=True)
	pd.DataFrame(rvalues_entropy).T.to_csv(os.path.join(data_path, "correlations_entropy.tsv"), sep="\t", header=True, index=True)
	print(failing_experiments)
