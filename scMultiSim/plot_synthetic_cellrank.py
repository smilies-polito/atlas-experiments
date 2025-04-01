import os
import itertools
import pandas as pd
from core.utils import simple_heatmap, simple_scatter, plot_branch_correlation, compute_correlation

if __name__=="__main__":
	
	# Declaration of path storing tsv results
	data_path = ... 
	
	# Results Folders 
	kl_path = os.path.join(data_path, "kl_results")
	entropy_path = os.path.join(data_path, "entropy_results")
	if not os.path.exists(kl_path):
		os.mkdir(kl_path)
	if not os.path.exists(entropy_path):
		os.mkdir(entropy_path)

	# Parameters 
	grid = {"diff_cif_fraction": [.1,.3,.5,.7,.9], "cif_sigma": [.1,.3,.5,.7,.9]}
	df = None
	model_list = ["multiomics", "rna"]
	branch= ... 

	# Read Run Results and create DataFrame
	for model in model_list: 
		for values in itertools.product(*grid.values()):
			diff_cif_fraction, cif_sigma = values
			path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_{model}")
			data = pd.read_csv(os.path.join(path, f"terminal.tsv"), sep="\t", index_col=0, header=0)
			data = data[["pseudotime", "entropy", "KL", "celltype"]]
			data["model"] = model
			data["diff_cif_fraction"] = diff_cif_fraction
			data["cif_sigma"] = cif_sigma 

			if df is None:
				df = data
			else:
				df = pd.concat((df, data), axis=0, ignore_index=True)


	# Compute kl-divergence and entropy correlation for every model and grid configuration parameters 
	for model in model_list:
		submodel = df[df["model"]==model]

		grouped_df = submodel[["diff_cif_fraction", "cif_sigma", "pseudotime", "entropy"]].groupby(by=["diff_cif_fraction", "cif_sigma"])
		values = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "entropy", return_pvalue=False).unstack(level="cif_sigma")
		pvalues = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "entropy").unstack(level="cif_sigma")
		simple_heatmap(values, (pvalues>=0.5).values, annot=True, save=True, saving_path=os.path.join(entropy_path, f"{model}_pearson_entropy.png"), cmap_range=(-1,1), xticklabels = values.columns, yticklabels=values.index, xlabel="cif_sigma", ylabel = "diff_cif_fraction", title= "Pearson Entropy-True Pseudotime")

		grouped_df = submodel[["diff_cif_fraction", "cif_sigma", "KL", "pseudotime"]].groupby(by=["diff_cif_fraction", "cif_sigma"])
		values = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "KL", return_pvalue=False).unstack(level="cif_sigma")
		pvalues = grouped_df.apply(compute_correlation, "pearson", "pseudotime", "KL").unstack(level="cif_sigma")
		simple_heatmap(values, (pvalues>=0.5).values, annot=True, save=True, saving_path=os.path.join(kl_path, f"{model}_pearson_kl.png"), cmap_range=(-1, 1), xticklabels = values.columns, yticklabels=values.index, xlabel="cif_sigma", ylabel = "diff_cif_fraction", title= "Pearson KL-True Pseudotime")

	# scatter plot models for every brach	
	df["model"] = pd.Categorical(df["model"])
	
	for values in itertools.product(*grid.values()):
		diff_cif_fraction, cif_sigma = values
		subdf = df[(df["diff_cif_fraction"] == diff_cif_fraction) & (df["cif_sigma"]==cif_sigma)]
		path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_multiomics")
		# entropy
		plot_branch_correlation(dataframe=subdf, key1="pseudotime", key2="entropy", group_key="celltype", color_key="model", branch=branch, save=True, saving_path = path, title=f"Scatter True Pseudotime-Entropy rd={diff_cif_fraction} sigma_cif={cif_sigma}", legend_loc = "lower center", legend_ncols=2, xlabel= "True Pseudotime", ylabel="Entropy", xlim=(0, 1.01), ylim=(0, subdf["entropy"].max() + .01))
		# kl 
		path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_rna")
		plot_branch_correlation(dataframe=subdf, key1="pseudotime", key2="KL", group_key="celltype", color_key="model", branch=branch, save=True, saving_path = path, title=f"Scatter True Pseudotime-KL rd={diff_cif_fraction} sigma_cif={cif_sigma}", legend_loc = "lower center", legend_ncols=2, xlabel= "True Pseudotime", ylabel="KL Divergence", xlim=(0, 1.01), ylim=(0, subdf["KL"].max()+0.01))

	
