import os
import pandas as pd
from core.metrics import plot_branch_correlation


if __name__=="__main__":

	diff_cif_fraction, sigma_cif = ...
	results_multiomics = pd.read_csv( ... , sep="\t", header=0, index_col=0)
	results_multiomics["model"] = "multiomics" 
	results_rna = pd.read_csv( ... , sep="\t", header=0, index_col=0)
	results_rna["model"] = "rna"

	print(results_multiomics.isna().any())
	print(results_rna.isna().any())

	branches = {}

	results = pd.concat((results_multiomics, results_rna), axis=0, ignore_index = True)
	results["model"] = pd.Categorical(results["model"])

	# entropy fixed terminal
	path = ...
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="true_pseudotime", key2="entropy_fixed", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - Entropy rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="entropy", xlim = (0,1.01), ylim= (0, results.entropy_fixed.max() + .01))
	path = ... 
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="true_pseudotime", key2="pseudotime_fixed", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - Pseudotime rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="palantir pseudotime", xlim = (0,1.01), ylim= (0, + .01))

	
	# entropy no terminal
	path = ... 
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="true_pseudotime", key2="entropy", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - Entropy rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="entropy", xlim = (0,1.01), ylim= (0, results.entropy.max() + .01))

	path = ... 
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="true_pseudotime", key2="pseudotime", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - Pseudotime rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="palantir pseudotime", xlim = (0,1.01), ylim= (0, 1.01))

