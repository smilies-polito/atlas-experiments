import os
import itertools
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from core.utils import simple_heatmap, simple_scatter, plot_branch_correlation

def get_group_correlation(group, method_key, key1, key2, return_pvalue:bool=True):
	method = pearsonr if method_key == "pearson" else spearmanr
	corr, pvalue = method(group[key1], group[key2])
	if return_pvalue:
		return pvalue
	else:
		return corr

if __name__=="__main__":
	data_path = "/Users/lrcq/Desktop/results_no_terminal"
	pseudotime_path = os.path.join(data_path, "pseudotime_results")
	entropy_path = os.path.join(data_path, "entropy_reults")
	if not os.path.exists(pseudotime_path):
		os.mkdir(pseudotime_path)
	if not os.path.exists(entropy_path):
		os.mkdir(entropy_path)

	grid = {"diff_cif_fraction": [.1,.3,.5,.7,.9], "cif_sigma": [.1,.3,.5,.7,.9]}
	df = None
	model_list = ["multiomics", "rna"]
	keys = [("rna:pseudotime", "palantir_pseudotime", "palantir_entropy", "rna:pop"), ("pseudotime", "palantir_pseudotime", "palantir_entropy", "pop")]
	branch= {"4-1": ["4_1"], "4-5-3": ["4_5", "5_3"], "4-5-2":["4_5", "5_2"]}

	for idx, model in enumerate(model_list):
		true_pseudotime_key, pseudotime_key, entropy_key, group_key = keys[idx]

		for values in itertools.product(*grid.values()):
			diff_cif_fraction, cif_sigma = values
			path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_{model}")
			data = pd.read_csv(os.path.join(path, f"results_{diff_cif_fraction}_{cif_sigma}.tsv"), sep="\t", index_col=0, header=0)
			data = data[["model", true_pseudotime_key, pseudotime_key, entropy_key, group_key]]
			data["diff_cif_fraction"] = diff_cif_fraction
			data["cif_sigma"] = cif_sigma 
			data.columns = ["model", "true_pseudotime", "palantir_pseudotime", "palantir_entropy", "celltype", "diff_cif_fraction", "cif_sigma"]

			if df is None:
				df = data
			else:
				df = pd.concat((df, data), axis=0, ignore_index=True)

	#heatmap for every model 
	for model in model_list:
		submodel = df[df["model"]==model]

		grouped_df = submodel[["diff_cif_fraction", "cif_sigma", "palantir_pseudotime", "true_pseudotime"]].groupby(by=["diff_cif_fraction", "cif_sigma"])
		values = grouped_df.apply(get_group_correlation, "pearson", "true_pseudotime", "palantir_pseudotime", return_pvalue=False).unstack(level="cif_sigma")
		pvalues = grouped_df.apply(get_group_correlation, "pearson", "true_pseudotime", "palantir_pseudotime").unstack(level="cif_sigma")
		simple_heatmap(values, (pvalues>=0.5).values, annot=True, save=True, saving_path=os.path.join(pseudotime_path, f"{model}_pearson_pseudotime.png"), cmap_range=(-1, 1), xticklabels = values.columns, yticklabels=values.index, xlabel="cif_sigma", ylabel = "diff_cif_fraction", title= "Pearson Pseudotime-True Pseudotime")

		grouped_df = submodel[["diff_cif_fraction", "cif_sigma", "palantir_entropy", "true_pseudotime"]].groupby(by=["diff_cif_fraction", "cif_sigma"])
		values = grouped_df.apply(get_group_correlation, "pearson", "true_pseudotime", "palantir_entropy", return_pvalue=False).unstack(level="cif_sigma")
		pvalues = grouped_df.apply(get_group_correlation, "pearson", "true_pseudotime", "palantir_entropy").unstack(level="cif_sigma")
		simple_heatmap(values, (pvalues>=0.5).values, annot=True, save=True, saving_path=os.path.join(entropy_path, f"{model}_pearson_entropy.png"), cmap_range=(-1, 1), xticklabels = values.columns, yticklabels=values.index, xlabel="cif_sigma", ylabel = "diff_cif_fraction", title= "Pearson Entropy-True Pseudotime")

	# scatter plot models for every brach	
	df["model"] = pd.Categorical(df["model"])
	
	for values in itertools.product(*grid.values()):
		diff_cif_fraction, cif_sigma = values
		subdf = df[(df["diff_cif_fraction"] == diff_cif_fraction) & (df["cif_sigma"]==cif_sigma)]
		path = os.path.join(data_path, f"{diff_cif_fraction}_{cif_sigma}_multiomics")
		plot_branch_correlation(dataframe=subdf, key1="true_pseudotime", key2="palantir_pseudotime", group_key="celltype", color_key="model", branch=branch, save=True, saving_path = path, title=f"Scatter True Pseudotime-Pseudotime rd={diff_cif_fraction} sigma_cif={cif_sigma}", legend_loc = "lower center", legend_ncols=2, xlabel= "True Pseudotime", ylabel="Palantir Pseudotime", xlim=(0, 1.01), ylim=(0, 1.01))
	plot_branch_correlation(dataframe=subdf, key1="true_pseudotime", key2="palantir_entropy", group_key="celltype", color_key="model", branch=branch, save=True, saving_path = path, title=f"Scatter True Pseudotime-Entropy rd={diff_cif_fraction} sigma_cif={cif_sigma}", legend_loc = "lower center", legend_ncols=2, xlabel= "True Pseudotime", ylabel="Palantir Entropy", xlim=(0, 1.01), ylim=(0, subdf["palantir_entropy"].max()+0.01))

