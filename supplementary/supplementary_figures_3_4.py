import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


SHORT_METRIC_RENOM = {"spearman_stat_pseudotime": "Spearman pseudo.",
				"kendall_stat_pseudotime": "Kendall pseudo.",	
				"temporal_state_score": "TSS",
				"fate_index_stat": "Fate Concentr.",
				"terminal_enrichment": "T.S. Enrichment",
				"terminal_silhouette_soft": "Silhouette - Soft",
				"terminal_silhouette_pse": "Silhouette - Pseudo.",
				"spearman_stat_SHE" : "Spearman SHE",
				"spearman_stat_KLD": "Spearman KL",
				"pearson_stat_SHE": "Pearson SHE",
				"pearson_stat_KLD": "Pearson KL",
				"totipotent": "JSD - T",
				"multipotent": "JSD - M",
				"committed": "JSD - C"
				}



if __name__== "__main__":
	algorithm = "palantir"
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

	# PLOT PSEUDOTIME
	if algorithm == "palantir":
		fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(12,6))
		for i, metric in enumerate(PSEUDOTIME_METRICS):	
			limit = METRIC_DICT[metric]
			title = METRIC_RENOM[metric]
			sub = success.loc[~success["fixed_terminal"], ["dataset_id", metric]]
			sns.boxplot(sub, x = "dataset_id", y=metric, ax=ax[i])
			ax[i].set_ylim(limit)
			ax[i].set_ylabel("")
			ax[i].set_xlabel("")
			ax[i].set_title(title)
		fig.savefig(os.path.join(output_path, f"{algorithm}_pseudotime.png"))
		plt.close()

	# PLOT TERMINAL STATE METRICS		
	fig, ax = plt.subplots(nrows = 2 , ncols = 2, figsize=(12,18))
	xidx, yidx = 0, 0
	for i, metric in enumerate(TEMPORAL_STATE_METRICS):
		yidx, xidx = i%2, i//2
		title = METRIC_RENOM[metric]
		limit = METRIC_DICT[metric]
		sub = success.loc[~success["fixed_terminal"], ["dataset_id", metric]]
		sns.boxplot(sub, x = "dataset_id", y=metric, ax=ax[xidx][yidx])
		ax[xidx][yidx].set_ylim(limit)
		ax[xidx][yidx].set_ylabel("")
		ax[xidx][yidx].set_xlabel("")
		ax[xidx][yidx].set_title(title)
	fig.savefig(os.path.join(output_path, f"{algorithm}_terminal_states.png"))
	plt.close()


	# PLOT TERMINAL STATE METRICS		
	fig, ax = plt.subplots(nrows = 4 , ncols = 2, figsize=(12,18))
	row, col = 0, 0
	for i, metric in enumerate(FATE_METRICS):
		col, row = i%2, i//2
		title = METRIC_RENOM[metric]
		limit = METRIC_DICT[metric]

		if metric in ["totipotent", "committed"]:
			dataset_condition = success["fixed_terminal"] 
		elif metric == "multipotent":
			dataset_condition = (success["tree"] == "five_branches") & (success["fixed_terminal"])
		else:
			dataset_condition = ~success["fixed_terminal"]

		sub = success.loc[dataset_condition, ["dataset_id", metric]]
		sns.boxplot(sub, x = "dataset_id", y=metric, ax=ax[row,col])
		ax[row,col].set_ylim(limit)
		ax[row,col].set_ylabel("")
		ax[row,col].set_xlabel("")
		ax[row,col].set_title(title)
	fig.savefig(os.path.join(output_path, f"{algorithm}_fates.png"))
	plt.close()

	
	# RADAR PLOT
	non_fixed_terminal_metrics = [m for m in METRIC_DICT.keys() if m not in ["multipotent", "totipotent", "committed", "ttp", "tsr", "ttc", "tts"]]
	fixed_terminal_metrics = ["multipotent", "totipotent", "committed"]
	

	summary_nonF = (success[~success["fixed_terminal"]].groupby("tree")[non_fixed_terminal_metrics]
											.agg( "median")
					)
	summary_nonF = (summary_nonF.stack(level=0)
								.rename_axis(index=["dataset", "metric"])
								.reset_index()
					)
	summary_F = (success[success["fixed_terminal"]].groupby("tree")[fixed_terminal_metrics]
											.agg("median")
				)
	summary_F = (summary_F.stack(level=0)
								.rename_axis(index=["dataset", "metric"])
								.reset_index()
					)
	summary = pd.concat((summary_nonF, summary_F))
	summary.columns = ["dataset", "metric", "median"]
	summary["metric"] = summary["metric"].map(SHORT_METRIC_RENOM)
	radar_df = summary.pivot(index="dataset", columns="metric", values="median")

	fig = go.Figure()
	for dataset in radar_df.index:
		values = radar_df.loc[dataset].tolist()
		metrics = radar_df.columns.tolist()
		values += [values[0]]
		metrics_closed = metrics + [metrics[0]]
		
		fig.add_trace(go.Scatterpolar(
							r = values,
							theta = metrics_closed,
							fill = "toself",
							name = dataset)
			)
	fig.update_layout(
			polar = dict(
						radialaxis = dict(visible = True, range = [-1,1]),
						angularaxis = dict(rotation= 90,
											direction = "clockwise",
											tickfont = dict(size=20))
					), 
			showlegend = True,
			legend = dict(font=dict(size=22)),
			width = 1000,
			height = 1000,
			margin = dict(l=200, r=150, t=80, b=80),
			template = "plotly_white")
		
	fig.write_image(os.path.join(output_path, f"{algorithm}_radar_plot.pdf"), 
					width = 1200, 
					height = 1200,
					scale = 3)


		

	
	

