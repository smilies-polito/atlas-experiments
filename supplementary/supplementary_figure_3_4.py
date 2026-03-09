import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

METRIC_DICT = {"spearman_stat_pseudotime": (-1,1),
				"kendall_stat_pseudotime": (-1,1),
				"temporal_state_score": (0,1),
				"tts": (0,1),
				"ttp": (0,1), 
				"tsr": (0,1), 
				"ttc": (0,1),
				"fate_index_stat": (-1,1),
				"terminal_enrichment": (-1,1),
				"terminal_silhouette_soft": (-1,1),
				"terminal_silhouette_pse": (-1,1), 
				"spearman_stat_SHE": (-1,1),
				"spearman_stat_KLD": (-1,1),
				"pearson_stat_SHE": (-1,1),
				"pearson_stat_KLD": (-1,1),
				"totipotent": (0, np.log(2)),
				"multipotent": (0, np.log(2)), 
				"committed": (0, np.log(2))
				}

PSEUDOTIME_METRICS = ["spearman_stat_pseudotime", "kendall_stat_pseudotime"]
TEMPORAL_STATE_METRICS = ["temporal_state_score", "terminal_enrichment", "terminal_silhouette_soft", "terminal_silhouette_pse"]
FATE_METRICS = ["fate_index_stat", "spearman_stat_KLD", "pearson_stat_KLD", "spearman_stat_SHE", "pearson_stat_SHE", "totipotent", "multipotent", "committed"]

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

if __name__== "__main__":
	algorithm = "pseudotime_kernel"
	output_path = os.path.join(os.getcwd(), "output", "simulations", algorithm)
	path = os.path.join(output_path, "results.csv")
	atlas = pd.read_csv(path)

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
	atlas["rd"] = atlas["rd"].astype(float)
	atlas["sigma"] = atlas["sigma"].astype(float)
	atlas["knn_rna"] = atlas["knn_rna"].astype(int)
	atlas["knn_activity"] = atlas["knn_activity"].astype(int)
	atlas["wnn"] = atlas["wnn"].astype(int)


	global_failure = atlas["failed"].mean() * 100
	success = atlas[~atlas["failed"]].copy()
	success["dataset_id"] = success["tree"].map({
												"three_branches": "3B",
												"five_branches": "5B"}) + "_" + success["fixed_terminal"].astype(str)

	nrows = 6
	if algorithm == "palantir":
		nrows += 1
	ncols = 2

	fig,ax = plt.subplots(nrows=nrows, 
							ncols=ncols,
							figsize = (14, 2.6*nrows),
							dpi = 300,
							constrained_layout=True)
	ax = ax.flatten()	
	idx = 0

	# PLOT PSEUDOTIME
	if algorithm == "palantir":
		for i, metric in enumerate(PSEUDOTIME_METRICS):	
			limit = METRIC_DICT[metric]
			title = METRIC_RENOM[metric]
			sub = success.loc[~success["fixed_terminal"], ["dataset_id", metric]]
			sns.boxplot(sub, x = "dataset_id", y=metric, ax=ax[idx])
			ax[idx].set_ylim(limit)
			ax[idx].set_ylabel("")
			ax[idx].set_xlabel("")
			ax[idx].set_title(title)
			idx += 1

	# PLOT TERMINAL STATE METRICS		
	for i, metric in enumerate(TEMPORAL_STATE_METRICS):
		title = METRIC_RENOM[metric]
		limit = METRIC_DICT[metric]
		sub = success.loc[~success["fixed_terminal"], ["dataset_id", metric]]
		sns.boxplot(sub, x = "dataset_id", y=metric, ax=ax[idx])
		ax[idx].set_ylim(limit)
		ax[idx].set_ylabel("")
		ax[idx].set_xlabel("")
		ax[idx].set_title(title)
		idx += 1


	# PLOT TERMINAL STATE METRICS		
	for i, metric in enumerate(FATE_METRICS):
		title = METRIC_RENOM[metric]
		limit = METRIC_DICT[metric]

		if metric in ["totipotent", "committed"]:
			dataset_condition = success["fixed_terminal"] 
		elif metric == "multipotent":
			dataset_condition = (success["tree"] == "five_branches") & (success["fixed_terminal"])
		else:
			dataset_condition = ~success["fixed_terminal"]

		sub = success.loc[dataset_condition, ["dataset_id", metric]]
		sns.boxplot(sub, x = "dataset_id", y=metric, ax=ax[idx])
		ax[idx].set_ylim(limit)
		ax[idx].set_ylabel("")
		ax[idx].set_xlabel("")
		ax[idx].set_title(title)
		idx += 1

	fig.savefig(os.path.join(output_path, f"metrics.png"))


	# METRICS CORRELATION		
	configs = ( success[["tree", "fixed_terminal"]]
					.drop_duplicates()
					.itertuples(index=False, name=None)
			)


	for tree, fixed_terminal in configs:
		metric_list = FATE_METRICS + TEMPORAL_STATE_METRICS 
		if algorithm == "palantir":
			metric_list = metric_list + PSEUDOTIME_METRICS
			
		sub = success.loc[(success["tree"]==tree) & (success["fixed_terminal"]==fixed_terminal), metric_list]
		corr = sub.corr()
		corr = corr.rename(index= METRIC_RENOM, columns = METRIC_RENOM)

		figure, ax = plt.subplots(figsize=(8, 12))
		sns.heatmap(corr, 
					annot=True, 
					center=0, 
					square=True,
					linewidths=0.2,
					vmin = -1,
					vmax= 1, 
					ax = ax)
		figure.savefig(os.path.join(output_path, f"correlation_{tree}_{fixed_terminal}.png"))




			

						
