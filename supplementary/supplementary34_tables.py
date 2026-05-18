import os
import argparse
import numpy as np 
import pandas as pd					
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


METRIC_LIST = [
		('spearman_stat_pseudotime', -1),
		('kendall_stat_pseudotime', -1),
		('temporal_state_score', 0),
		('pearson_stat_KLD', -1),
		('pearson_stat_SHE', 1),
		('spearman_stat_KLD', -1),
		('spearman_stat_SHE', 1),
		('fate_index_stat', -1),
		('terminal_silhouette_soft', -1),
		('terminal_silhouette_pse', -1),
		('terminal_enrichment', -1),
		('committed', np.sqrt(np.log(2))),
		('multipotent', np.sqrt(np.log(2))),
		('totipotent', np.sqrt(np.log(2)))]

def _compute_robustness(df, 
						algorithm:str,
						success_col:str,
						ci_level:float= 0.95):
	dataset_cols = ["tree", "fixed_terminal"]
	alpha = 1 - ci_level
	results = []
	for (tree, fixed_terminal), sub in df.groupby(dataset_cols):
		successes = sub[success_col].sum() if len(sub) > 0 else 0
		runs = len(sub)
		p_hat = successes/runs if runs > 0 else np.nan
		ci_low, ci_high = proportion_confint(count=successes,
												nobs = runs,
												alpha = alpha, 
												method = "wilson")
		results.append({"algorithm" : algorithm,
						"tree": tree,
						"fixed_terminal": fixed_terminal,
						"total_runs": runs,
						"valid_runs": successes,
						"p_hat": p_hat, 
						"ci_high": ci_high,
						"ci_low": ci_low})
	return pd.DataFrame(results)
						
						
	

def compare_algorithms(A, R, rng,
						metric_col:str, 
						success_col: str,
						worst_value: float,
						ci_level: float = 0.95,
						n_boot: int = 2000):

	dataset_cols = ["tree", "fixed_terminal"]
	common_hp = ["rd", "sigma", "knn_rna"]
	alpha = 1 - ci_level

	A = A.copy()
	R = R.copy()
	A[metric_col] = A[metric_col].fillna(worst_value)
	R[metric_col] = R[metric_col].fillna(worst_value)

	RG = R.groupby(dataset_cols + common_hp).agg(
												mean_metric_R = (metric_col, "mean"),
												success_rate_R = (success_col, "mean"),
												n_runs_R = (metric_col, "count")
											).reset_index()

	AG = A.groupby(dataset_cols + common_hp).agg(
												mean_metric_A = (metric_col, "mean"),
												success_rate_A = (success_col, "mean"),
												n_runs_A = (metric_col, "count")
											).reset_index()
	
	merged = pd.merge(AG, RG, on = dataset_cols + common_hp, how="inner")  
	

	# performance condizionata 
	performance_results = []
	merged["delta"] = merged["mean_metric_A"] - merged["mean_metric_R"]
	
	results = []
	
	for (tree, fixed_terminal), sub in merged.groupby(dataset_cols):
		deltas = sub["delta"].values
		n = len(deltas)

		if n==0:
			continue

		mean_delta = np.mean(deltas)
		median_delta = np.median(deltas)
		p_hat = np.mean(deltas > 0)
		
		boot_means = []
		boot_medians = []

		for _ in range(n_boot):
			sample = rng.choice(deltas, size=n, replace=True)
			boot_means.append(np.mean(sample))
			boot_medians.append(np.median(sample))
	
		ci_mean = np.percentile(boot_means, [100 * alpha/2, 100*(1-alpha)/2])
		ci_median = np.percentile(boot_medians, [100 * alpha/2, 100*(1-alpha)/2])
		ci_low, ci_high = proportion_confint(count = np.sum(deltas>0), 
												nobs = n, 
												alpha = alpha,
												method = "wilson")

		performance_results.append({
									"metric": metric_col,
									"tree": tree,
									"fixed_terminal": fixed_terminal,
									"mean_delta": mean_delta,
									"mean_delta_CI_low": ci_mean[0],
									"mean_delta_CI_high": ci_mean[1],
									"median_delta": median_delta,
									"median_delta_CI_low": ci_median[0],
									"median_delta_CI_high": ci_median[1],
									"p_A_better": p_hat,
									"p_A_better_CI_low": ci_low, 
									"p_A_better_CI_high": ci_high
					})

	performance = pd.DataFrame(performance_results)
	return (robustness, performance)
								
		
		

			



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="palantir")
    args= parser.parse_args()
    algorithm = args.algorithm 

	rng = np.random.default_rng(42)

	data_path = os.path.join(os.getcwd(), "output", "simulations")	
	rna = pd.read_csv(os.path.join(data_path, f"{algorithm}_rna", "results.csv"), sep=",", header = 0, index_col = None)
	atlas = pd.read_csv(os.path.join(data_path, f"{algorithm}", "results.csv"), sep=",", header = 0, index_col = None)
	
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
	atlas["failed"] = ~atlas["failed"]

	# parse rna params
	main_split = rna["code"].str.split("_", expand=True)

	rna["tree"] = main_split[0] + "_" + main_split[1]
	rna["fixed_terminal"] = main_split[2]
	rna["rd"] = main_split[3]
	rna["sigma"] = main_split[4]
	rna["knn_rna"] = main_split[5]

	rna["fixed_terminal"] = rna["fixed_terminal"].map({"True": True, "False": False})
	rna["rd"] = rna["rd"].astype(float)
	rna["sigma"] = rna["sigma"].astype(float)
	rna["knn_rna"] = rna["knn_rna"].astype(int)
	rna["failed"] = ~rna["failed"]


	wholeP, wholeR = None, None
	ci_level = 0.95


	A_robustness = _compute_robustness(atlas,
										algorithm = "atlas", 
										success_col = "failed")
	R_robustness = _compute_robustness(rna,
										algorithm = "atlas", 
										success_col = "failed")
	robustness = pd.concat((A_robustness, R_robustness), axis=0)

	
	for metric, worst_value in METRIC_LIST:
		robustness, results =  compare_algorithms(A=atlas, R=rna, rng= rng,
					worst_value = worst_value,
					metric_col= metric,
					success_col= "failed",
					ci_level = ci_level)

		wholeP = results if wholeP is None else pd.concat([wholeP, results], axis=0)


	wholeP[f"Delta mean ({ci_level * 100:.0f} %)"] = (
    		wholeP["mean_delta"].map("{:.3f}".format)
   			+ " ("
   			+ wholeP["mean_delta_CI_low"].map("{:.3f}".format)
    		+ ", "
    		+ wholeP["mean_delta_CI_high"].map("{:.3f}".format)
		    + ")"
	)

	wholeP[f"Delta median ({ci_level * 100:.0f} %)"] = (
    		wholeP["median_delta"].map("{:.3f}".format) 
   			+ " ("
   			+ wholeP["median_delta_CI_low"].map("{:.3f}".format)
    		+ ", "
    		+ wholeP["median_delta_CI_high"].map("{:.3f}".format)
		    + ")"
	)

	wholeP[f"P(ATLAS > baseline) ({ci_level * 100:.0f} %)"] = (
    		wholeP["p_A_better"].map("{:.3f}".format)
   			+ " ("
   			+ wholeP["p_A_better_CI_low"].map("{:.3f}".format)
    		+ ", "
    		+ wholeP["p_A_better_CI_high"].map("{:.3f}".format)
		    + ")"
	)

	cols_to_retain = ["metric", "tree", "fixed_terminal", f"Delta mean ({ci_level * 100:.0f} %)", f"Delta median ({ci_level * 100:.0f} %)", f"P(ATLAS > baseline) ({ci_level * 100:.0f} %)"]

	performance_path = os.path.join(data_path, "performance_pseudotime_kernel.tsv")
	robustness_path = os.path.join(data_path, "robustness_pseudotime_kernel.tsv")
	wholeP[cols_to_retain].to_csv(performance_path, header=True, index=False, sep="\t")	
	robustness.to_csv(robustness_path, header=True, index=False, sep="\t")	
	
	
			

		

	
	
