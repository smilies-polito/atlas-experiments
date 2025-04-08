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

	branches = { ... }
	results = pd.concat((results_multiomics, results_rna), axis=0, ignore_index = True)
	results["model"] = pd.Categorical(results["model"])

	# entropy fixed terminal
	path = ... 
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="pseudotime", key2="fixed_entropy", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - Entropy rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="entropy", xlim = (0,1.01), ylim= (0, results.fixed_entropy.max() + .01))

	# kl fixed terminal
	path = ... 
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="pseudotime", key2="fixed_kl", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - KL-div rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="kl-divergence", xlim = (0,1.01), ylim= (0, results.fixed_kl.max() + .01))
	
	# entropy no terminal
	path = ... 
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="pseudotime", key2="entropy", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - Entropy rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="entropy", xlim = (0,1.01), ylim= (0, results.entropy.max() + .01))

	# kl no terminal
	path = ...
	if not os.path.exists(path):
		os.mkdir(path)
	plot_branch_correlation(dataframe=results, key1="pseudotime", key2="kl", group_key="celltype", color_key="model", branch = branches, save=True, saving_path = path, title=f"Scatter True Pseudotime - KL-div rd={diff_cif_fraction} sigma_cif={sigma_cif}", legend_loc="lower center", legend_ncols = 2, xlabel= "true pseudotime", ylabel="kl-divergence", xlim = (0,1.01), ylim= (0, results.kl.max() + .01))

