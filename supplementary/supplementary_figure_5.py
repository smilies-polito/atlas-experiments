import os
import seaborn as sns
import numpy as np 
import pandas as pd					
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from statsmodels.stats.proportion import proportion_confint


METRIC_LIST = [
		'spearman_stat_pseudotime',
		'kendall_stat_pseudotime',
		'temporal_state_score',
		'pearson_stat_KLD',
		'pearson_stat_SHE',
		'spearman_stat_KLD',
		'spearman_stat_SHE',
		'fate_index_stat',
		'terminal_silhouette_soft',
		'terminal_silhouette_pse',
		'terminal_enrichment']

FIXED_TERMINAL =[
		'committed',
		'multipotent',
		'totipotent']


hyperparams = [
		'rd',
		'sigma',
		'knn_rna',
		'knn_activity',
		'wnn'
	]

TREE_DICT = {"five_branches": "Five Branches Differentiation Tree",
			"three_branches": "Three Branches Differentiation Tree"}


METRIC_RENOM = {"spearman_stat_pseudotime": "Spearman corr. pseudotime",
				"kendall_stat_pseudotime": "Kendall corr. pseudotime",	
				"temporal_state_score": "Temporal State Score",
				"tts": "Terminal Timing Score",
				"ttp": "Terminal Temporal Precision",
				"tsr": "Terminal State Recall",
				"ttc": "Terminal Temporal Concentration",
				"fate_index_stat": "Fate Concentration Index",
				"terminal_enrichment": "Terminal State Enrichment",
				"terminal_silhouette_soft": "Soft Terminal State Silhouette",
				"terminal_silhouette_pse": "Terminal State Silhouette - pseudotime",
				"spearman_stat_SHE": "Spearman corr. shannon entropy",
				"spearman_stat_KLD": "Spearman corr. KL-divergence",
				"pearson_stat_SHE": "Pearson corr. shannon entropy",
				"pearson_stat_KLD": "Pearson corr. KL-divergence",
				"totipotent": "JSD - totipotent",
				"multipotent": "JSD - multipotent",
				"committed": "JSD- committed"
				}

								
if __name__ == "__main__":
	rng = np.random.default_rng(42)
	algorithm = "pseudotime_kernel"

	data_path = os.path.join(os.getcwd(), "output", "simulations")	
	atlas = pd.read_csv(os.path.join(data_path, algorithm, "results.csv"), sep=",", header = 0, index_col = None)
	
	# parse atlas params
	main_split = atlas["code"].str.split("_", expand=True)

	atlas["tree"] = main_split[0] + "_" + main_split[1]
	atlas["fixed_terminal"] = main_split[2]
	atlas["rd"] = main_split[3]
	atlas["sigma"] = main_split[4]

	knn_split = main_split[5].str.split(":", expand=True)
	atlas["knn_rna"] = knn_split[0]
	atlas["knn_activity"] = knn_split[1]
	atlas["wnn"] = knn_split[2]

	atlas["fixed_terminal"] = atlas["fixed_terminal"].map({"True": True, "False": False})
	atlas["rd"] = atlas["rd"].astype("category")
	atlas["sigma"] = atlas["sigma"].astype("category")
	atlas["knn_rna"] = atlas["knn_rna"].astype("category")
	atlas["knn_activity"] = atlas["knn_activity"].astype("category")
	atlas["wnn"] = atlas["wnn"].astype("category")
	atlas["failed"] = ~atlas["failed"]
	
	atlas = atlas[atlas["failed"]] #removing non available 

	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(19,10))
	vmin, vmax = -1, 1
	cmap = "coolwarm"

	mean_correlation = []
	for i, tree in enumerate(["three_branches", "five_branches"]):
		sub_false = atlas[(atlas["tree"] == tree) & (atlas["fixed_terminal"]==False)]
		sub_false_dummies = pd.get_dummies(sub_false[hyperparams])
		sub_false = pd.concat([sub_false_dummies, sub_false[METRIC_LIST]], axis=1)
		corr_false = sub_false.corr(method="pearson")
		corr_hp_false = corr_false.loc[sub_false_dummies.columns, METRIC_LIST]

		sub_true = atlas[(atlas["tree"] == tree) & (atlas["fixed_terminal"])]	
		sub_true_dummies = pd.get_dummies(sub_true[hyperparams])
		sub_true = pd.concat([sub_true_dummies, sub_true[FIXED_TERMINAL]], axis=1)
		corr_true = sub_true.corr(method="pearson")
		corr_hp_true = corr_true.loc[sub_true_dummies.columns, FIXED_TERMINAL]

		corr = pd.merge(corr_hp_false, corr_hp_true, left_index= True, right_index=True, how="left")

		simulation_params = [i for i in corr.index.tolist() if i.startswith(("rd", "sigma"))]
		knn_params = [i for i in corr.index.tolist() if i.startswith(("knn_rna", "knn_activity", "wnn"))]
		mean_s_corr, mean_k_corr = corr.loc[simulation_params, :].mean(axis=None), corr.loc[knn_params, :].mean(axis=None)
		mean_correlation.append({
								"tree": tree,
								"avg_simulation_corr": mean_s_corr,
								"avg_knn_corr": mean_k_corr})

		corr = corr.rename(columns = METRIC_RENOM)
		sns.heatmap(corr, 
					ax=ax[i], 
					vmin = vmin,
					vmax = vmax,
					cbar = False,
					cmap = cmap,
					annot= True)
		ax[i].set_ylabel(tree)

	ax[0].set_xticklabels([])
	ax[1].tick_params(axis="x", rotation=15)


	sm = ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
	sm.set_array([])
	fig.colorbar(sm, ax=ax, location="right", shrink=0.8)

	plt.savefig(os.path.join(data_path, f"{algorithm}_hyperparameters_correlation.png"))

	mean_correlation = pd.DataFrame(mean_correlation)
	print(mean_correlation.round(3))
		
		

		
		
