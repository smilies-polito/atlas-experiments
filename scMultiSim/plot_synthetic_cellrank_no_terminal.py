import os
import muon as mu
import itertools
import pandas as pd
from core.utils import simple_heatmap, simple_scatter, plot_branch_correlation, compute_correlation


if __name__=="__main__":
	# Declare data path 
	data_path = ... 
	# Declare paths for results
	kl_path = os.path.join(data_path, "kl_results")
	entropy_path = os.path.join(data_path, "entropy_results")
	if not os.path.exists(kl_path):
		os.mkdir(kl_path)
	if not os.path.exists(entropy_path):
		os.mkdir(entropy_path)

	# Parameters
	grid = {"diff_cif_fraction": [.1,.3,.5,.7,.9], "cif_sigma": [.1,.3,.5,.7,.9], "n_states":[3,4,5,6,7,8]}
	df = None
	model_list = ["multiomics", "rna"]
	branch= ... 

	# Store all results into a single dataframe
	for model in model_list: 
		for values in itertools.product(*grid.values()):
			diff_cif_fraction, cif_sigma, nstates  = values
			path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_{model}")	
			muData = mu.read_h5mu(os.path.join(os.getcwd(), "phyla5", f"{diff_cif_fraction}_{cif_sigma}_data.h5mu"))
			path = os.path.join(path, f"terminal_{nstates}.tsv")
			if not os.path.exists(path):
				continue
			data = pd.read_csv(os.path.join(path), sep="\t", index_col=0, header=0)
			data = data[["entropy", "KL"]]
			data["model"] = model
			data["diff_cif_fraction"] = diff_cif_fraction
			data["cif_sigma"] = cif_sigma 
			data["celltype"] = muData.obs["rna:pop"].copy()	
			data["pseudotime"]= muData.obs["rna:pseudotime"].copy()
			data["nstates"] = nstates

			if df is None:
				df = data
			else:
				df = pd.concat((df, data), axis=0, ignore_index=True)
	
	#heatmap for every model 
	for model in model_list:
		for nstates in grid["n_states"]:
			submodel = df[(df["model"]==model) & (df["nstates"]==nstates)]
			if submodel.empty or submodel.isna().any().sum()>0:
				continue
		
			grouped_df = submodel[["diff_cif_fraction", "cif_sigma", "pseudotime", "entropy"]].groupby(by=["diff_cif_fraction", "cif_sigma"])
			values = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "entropy", return_pvalue=False).unstack(level="cif_sigma")
			pvalues = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "entropy").unstack(level="cif_sigma")
			simple_heatmap(values, (pvalues>=0.5).values, annot=True, save=True, saving_path=os.path.join(entropy_path, f"{model}_pearson_entropy_{nstates}.png"), cmap_range=(-1,1), xticklabels = values.columns, yticklabels=values.index, xlabel="cif_sigma", ylabel = "diff_cif_fraction", title= f"Pearson Entropy-True Pseudotime {nstates}")

			grouped_df = submodel[["diff_cif_fraction", "cif_sigma", "KL", "pseudotime"]].groupby(by=["diff_cif_fraction", "cif_sigma"])
			values = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "KL", return_pvalue=False).unstack(level="cif_sigma")
			pvalues = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "KL").unstack(level="cif_sigma")
			simple_heatmap(values, (pvalues>=0.5).values, annot=True, save=True, saving_path=os.path.join(kl_path, f"{model}_pearson_kl_{nstates}.png"), cmap_range=(-1, 1), xticklabels = values.columns, yticklabels=values.index, xlabel="cif_sigma", ylabel = "diff_cif_fraction", title= f"Pearson KL-True Pseudotime {nstates}")

	# scatter plot models for every brach	
	df["model"] = pd.Categorical(df["model"])
	
	for values in itertools.product(*grid.values()):
		diff_cif_fraction, cif_sigma, n_states = values
		subdf = df[(df["diff_cif_fraction"] == diff_cif_fraction) & (df["cif_sigma"]==cif_sigma) & (df["nstates"]==nstates)]
		if subdf.empty: #no configuration with diff_cif_fraction, cif_sigma and nstates 
			continue
		if subdf.KL.isna().sum()>0 or subdf.entropy.isna().sum()>0: # there are NaNs 
			print("KL or entropy void")
			continue

		path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_multiomics_{nstates}")
		if not os.path.exists(path):
			os.mkdir(path)
		# entropy
		plot_branch_correlation(dataframe=subdf, key1="pseudotime", key2="entropy", group_key="celltype", color_key="model", branch=branch, save=True, saving_path = path, title=f"{nstates} Scatter True Pseudotime-Entropy rd={diff_cif_fraction} sigma_cif={cif_sigma}", legend_loc = "lower center", legend_ncols=2, xlabel= "True Pseudotime", ylabel="Entropy", xlim=(0, 1.01), ylim=(0, subdf["entropy"].max() + .01))
		# kl 
		path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_rna_{nstates}")
		if not os.path.exists(path):
			os.mkdir(path)
		plot_branch_correlation(dataframe=subdf, key1="pseudotime", key2="KL", group_key="celltype", color_key="model", branch=branch, save=True, saving_path = path, title=f"{nstates} Scatter True Pseudotime-KL rd={diff_cif_fraction} sigma_cif={cif_sigma}", legend_loc = "lower center", legend_ncols=2, xlabel= "True Pseudotime", ylabel="KL Divergence", xlim=(0, 1.01), ylim=(0, subdf["KL"].max()+0.01))

